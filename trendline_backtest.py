import pandas as pd
import numpy as np
import backtrader as bt
from trendline_automation import TrendlineOptimizer


def calculate_atr(high, low, close, period=14):
    """
    计算真实波幅均值 (ATR)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR 周期
        
    Returns:
        ATR 序列
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def preprocess_data(file_path, lookback=30):
    """
    预处理数据，使用 TrendlineOptimizer 计算趋势线特征

    Args:
        file_path: CSV 文件路径
        lookback: TrendlineOptimizer 的 lookback 参数

    Returns:
        包含趋势线特征的 DataFrame
    """
    df = pd.read_csv(file_path)

    required_cols = ['date', 'close', 'open', 'high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入数据缺失必需列: '{col}'")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    optimizer = TrendlineOptimizer(lookback=lookback)
    df_with_features = optimizer.fit_transform(df)

    return df_with_features


def generate_signals(df, atr_period=14, atr_multiplier=0.5):
    """
    基于趋势线特征生成交易信号
    
    Args:
        df: 包含趋势线特征的 DataFrame
        atr_period: ATR 计算周期
        atr_multiplier: ATR 乘数，用于确定突破阈值
        
    Returns:
        包含交易信号的 DataFrame
    """
    # 创建副本以避免修改原始 DataFrame
    df = df.copy()
    
    # 检查必需列
    required_cols = ['resist_pivot_val_real', 'resist_slope_logspace', 'resist_pivot_abs',
                    'support_pivot_val_real', 'support_slope_logspace', 'support_pivot_abs']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺失必需列: {missing_cols}")
    
    # 计算 ATR
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    
    # 计算突破阈值
    breakout_threshold = atr * atr_multiplier
    
    # 初始化信号列
    df['long_signal'] = 0
    df['exit_signal'] = 0
    
    # 向量化生成做多信号
    # 检查有效的阻力线数据
    valid_resist_mask = (
        ~pd.isna(df['resist_pivot_val_real']) & 
        ~pd.isna(df['resist_slope_logspace']) & 
        (df['resist_pivot_abs'] >= 0)
    )
    
    # 计算阻力线值（对数空间）
    resist_pivot_val = df['resist_pivot_val_real'].clip(lower=1e-10)  # 防止日志为零或负数
    resist_line_value_log = np.log(resist_pivot_val) + \
                           df['resist_slope_logspace'] * (np.arange(len(df)) - df['resist_pivot_abs'])
    resist_line_value = np.exp(resist_line_value_log)
    
    # 检查突破条件
    threshold = breakout_threshold.fillna(0)
    long_signal_condition = (df['close'] > resist_line_value + threshold) & valid_resist_mask
    df.loc[long_signal_condition, 'long_signal'] = 1
    
    # 向量化生成平仓信号
    # 检查有效的支撑线数据
    valid_support_mask = (
        ~pd.isna(df['support_pivot_val_real']) & 
        ~pd.isna(df['support_slope_logspace']) & 
        (df['support_pivot_abs'] >= 0)
    )
    
    # 计算支撑线值（对数空间）
    support_pivot_val = df['support_pivot_val_real'].clip(lower=1e-10)  # 防止日志为零或负数
    support_line_value_log = np.log(support_pivot_val) + \
                            df['support_slope_logspace'] * (np.arange(len(df)) - df['support_pivot_abs'])
    support_line_value = np.exp(support_line_value_log)
    
    # 检查跌破条件
    exit_signal_condition = (df['close'] < support_line_value - threshold) & valid_support_mask
    df.loc[exit_signal_condition, 'exit_signal'] = 1
    
    return df


class SignalData(bt.feeds.PandasData):
    """
    自定义 backtrader 数据类，支持信号列
    """
    lines = ('long_signal', 'exit_signal',)
    params = (
        ('long_signal', -1),
        ('exit_signal', -1),
    )


class TrendlineBreakoutStrategy(bt.Strategy):
    """
    趋势线突破策略
    
    做多条件：价格向上突破下降趋势线（阻力线）
    平仓条件：价格跌破上升趋势线（支撑线）
    """
    
    def __init__(self):
        # 获取信号数据
        self.long_signal = self.data.long_signal
        self.exit_signal = self.data.exit_signal
        
    def next(self):
        # 如果没有持仓且出现做多信号，则买入
        if self.long_signal[0] == 1 and not self.position:
            # 计算可买入数量（使用全部资金）
            cash = self.broker.getcash()
            price = self.data.close[0]
            size = cash / price
            self.buy(size=size)
            
        # 如果有持仓且出现平仓信号，则卖出
        elif self.exit_signal[0] == 1 and self.position:
            self.sell(size=self.position.size)


def run_backtest(df, initial_cash=10000):
    """
    执行回测
    
    Args:
        df: 包含交易信号的 DataFrame
        initial_cash: 初始资金
        
    Returns:
        回测结果
    """
    # 检查必需列
    required_cols = ['open', 'high', 'low', 'close', 'long_signal', 'exit_signal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺失必需列: {missing_cols}")
    
    # 创建 cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(TrendlineBreakoutStrategy)
    
    # 准备数据
    # 添加额外列供 backtrader 使用
    df_bt = df[['open', 'high', 'low', 'close', 'long_signal', 'exit_signal']].copy()
    
    # 创建数据源
    data = SignalData(
        dataname=df_bt,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume=None,
        openinterest=-1
    )
    
    # 添加数据
    cerebro.adddata(data)
    
    # 设置初始资金
    cerebro.broker.setcash(initial_cash)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行回测
    results = cerebro.run()
    
    return results, cerebro


def print_results(results, cerebro):
    """
    打印回测结果
    
    Args:
        results: 回测结果
        cerebro: cerebro 引擎
    """
    # 检查结果是否为空
    if not results:
        raise ValueError("回测结果为空")
    
    strategy = results[0]
    
    # 获取分析结果
    returns_analysis = strategy.analyzers.returns.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    trades_analysis = strategy.analyzers.trades.get_analysis()
    
    # 打印结果
    print("\n" + "="*50)
    print("回测结果")
    print("="*50)
    
    # 基本信息
    print(f"初始资金: {cerebro.broker.startingcash:,.2f}")
    print(f"最终资金: {cerebro.broker.getvalue():,.2f}")
    
    # 收益率
    total_return = returns_analysis.get('rtot', 0)
    print(f"总收益率: {total_return*100:.2f}%")
    
    # 最大回撤
    max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0)
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    # 夏普比率
    sharpe_ratio = sharpe_analysis.get('sharperatio', 0)
    if sharpe_ratio is not None:
        print(f"夏普比率: {sharpe_ratio:.4f}")
    else:
        print("夏普比率: N/A")
    
    # 交易统计
    total_trades = trades_analysis.get('total', {}).get('total', 0)
    won_trades = trades_analysis.get('won', {}).get('total', 0)
    lost_trades = trades_analysis.get('lost', {}).get('total', 0)
    
    print(f"总交易次数: {total_trades}")
    print(f"盈利交易次数: {won_trades}")
    print(f"亏损交易次数: {lost_trades}")
    
    if total_trades > 0:
        win_rate = won_trades / total_trades * 100
        print(f"胜率: {win_rate:.2f}%")
    
    print("="*50)


if __name__ == "__main__":
    try:
        print("开始测试信号生成函数...")
        
        # 测试数据预处理
        print("1. 测试数据预处理...")
        df = preprocess_data("BTCUSDT3600.csv")
        print(f"   数据形状: {df.shape}")
        
        # 测试信号生成
        print("2. 测试信号生成...")
        df_with_signals = generate_signals(df)
        
        long_signals = df_with_signals['long_signal'].sum()
        exit_signals = df_with_signals['exit_signal'].sum()
        
        print(f"   信号生成成功")
        print(f"   做多信号数量: {long_signals}")
        print(f"   平仓信号数量: {exit_signals}")
        
        # 验证信号列存在
        assert 'long_signal' in df_with_signals.columns, "缺少 long_signal 列"
        assert 'exit_signal' in df_with_signals.columns, "缺少 exit_signal 列"
        
        print("所有测试通过！")
        
        # 测试回测执行函数
        print("\n3. 测试回测执行函数...")
        try:
            results, cerebro = run_backtest(df_with_signals)
            print("   回测执行成功")
            print(f"   最终资金: {cerebro.broker.getvalue():,.2f}")
        except Exception as e:
            print(f"   回测执行失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试结果输出函数
        print("\n4. 测试结果输出函数...")
        try:
            print_results(results, cerebro)
            print("结果输出成功")
        except Exception as e:
            print(f"结果输出失败: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
