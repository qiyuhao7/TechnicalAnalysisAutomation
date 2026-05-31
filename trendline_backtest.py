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
    平仓条件：价格跌破上升趋势线（支撑线）或触发 ATR 止损
    """
    
    # 策略参数
    params = (
        ('atr_period', 14),  # ATR 周期
        ('atr_multiplier', 2.0),  # ATR 乘数，用于计算止损距离
    )
    
    def __init__(self):
        # 获取信号数据
        self.long_signal = self.data.long_signal
        self.exit_signal = self.data.exit_signal
        
        # 计算 ATR
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # 记录交易信息
        self.trade_records = []
        self.buy_prices = []
        self.buy_dates = []
        self.sell_prices = []
        self.sell_dates = []
        
        # 止损相关变量
        self.stop_loss_price = None
        self.entry_atr = None
        
    def next(self):
        # 如果没有持仓且出现做多信号，则买入
        if self.long_signal[0] == 1 and not self.position:
            # 计算可买入数量（使用全部资金）
            cash = self.broker.getcash()
            price = self.data.close[0]
            size = cash / price
            self.buy(size=size)
            
            # 记录买入信息
            self.buy_prices.append(price)
            self.buy_dates.append(self.data.datetime.date(0))
            
            # 设置 ATR 止损
            self.entry_atr = self.atr[0]
            self.stop_loss_price = price - (self.params.atr_multiplier * self.entry_atr)
            
        # 如果有持仓，检查是否需要卖出
        elif self.position:
            # 检查是否触发 ATR 止损
            if self.stop_loss_price is not None and self.data.close[0] < self.stop_loss_price:
                price = self.data.close[0]
                self.sell(size=self.position.size)
                
                # 记录卖出信息
                self.sell_prices.append(price)
                self.sell_dates.append(self.data.datetime.date(0))
                
                # 重置止损变量
                self.stop_loss_price = None
                self.entry_atr = None
                
            # 检查是否出现趋势线平仓信号
            elif self.exit_signal[0] == 1:
                price = self.data.close[0]
                self.sell(size=self.position.size)
                
                # 记录卖出信息
                self.sell_prices.append(price)
                self.sell_dates.append(self.data.datetime.date(0))
                
                # 重置止损变量
                self.stop_loss_price = None
                self.entry_atr = None


def run_backtest(df, initial_cash=10000, atr_multiplier=2.0):
    """
    执行回测
    
    Args:
        df: 包含交易信号的 DataFrame
        initial_cash: 初始资金
        atr_multiplier: ATR 乘数，用于计算止损距离
        
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
    cerebro.addstrategy(TrendlineBreakoutStrategy, atr_multiplier=atr_multiplier)
    
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
    
    # 回撤详情
    max_drawdown_len = drawdown_analysis.get('max', {}).get('len', 0)
    max_drawdown_money = drawdown_analysis.get('max', {}).get('moneydown', 0)
    print(f"最大回撤持续时间: {max_drawdown_len} 个周期")
    print(f"最大回撤金额: {max_drawdown_money:,.2f}")
    
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
        
        # 计算平均盈利和平均亏损
        won_total = trades_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        lost_total = trades_analysis.get('lost', {}).get('pnl', {}).get('total', 0)
        
        if won_trades > 0:
            avg_won = won_total / won_trades
            print(f"平均盈利: {avg_won:,.2f}")
        
        if lost_trades > 0:
            avg_lost = lost_total / lost_trades
            print(f"平均亏损: {avg_lost:,.2f}")
        
        # 盈利因子
        if lost_total != 0:
            profit_factor = abs(won_total / lost_total)
            print(f"盈利因子: {profit_factor:.2f}")
        
        # 最大连续亏损
        max_consecutive_loss = trades_analysis.get('lost', {}).get('streak', {}).get('max', 0)
        print(f"最大连续亏损次数: {max_consecutive_loss}")
    
    print("="*50)


def analyze_trades(results, cerebro):
    """
    分析交易情况，找出回撤原因
    
    Args:
        results: 回测结果
        cerebro: cerebro 引擎
    """
    strategy = results[0]
    
    # 获取交易记录
    buy_prices = strategy.buy_prices
    sell_prices = strategy.sell_prices
    buy_dates = strategy.buy_dates
    sell_dates = strategy.sell_dates
    
    print("\n" + "="*50)
    print("交易分析")
    print("="*50)
    
    # 计算每笔交易的盈亏
    trade_results = []
    for i in range(min(len(buy_prices), len(sell_prices))):
        buy_price = buy_prices[i]
        sell_price = sell_prices[i]
        pnl = (sell_price - buy_price) / buy_price * 100
        trade_results.append({
            'buy_date': buy_dates[i],
            'sell_date': sell_dates[i],
            'buy_price': buy_price,
            'sell_price': sell_price,
            'pnl_percent': pnl
        })
    
    # 按盈亏排序
    trade_results.sort(key=lambda x: x['pnl_percent'])
    
    print(f"\n总交易次数: {len(trade_results)}")
    
    # 显示亏损最大的10笔交易
    print("\n亏损最大的10笔交易:")
    print("-" * 80)
    print(f"{'买入日期':<12} {'卖出日期':<12} {'买入价':<12} {'卖出价':<12} {'盈亏%':<10}")
    print("-" * 80)
    
    for trade in trade_results[:10]:
        print(f"{trade['buy_date']:<12} {trade['sell_date']:<12} "
              f"{trade['buy_price']:<12.2f} {trade['sell_price']:<12.2f} "
              f"{trade['pnl_percent']:<10.2f}%")
    
    # 显示盈利最大的10笔交易
    print("\n盈利最大的10笔交易:")
    print("-" * 80)
    print(f"{'买入日期':<12} {'卖出日期':<12} {'买入价':<12} {'卖出价':<12} {'盈亏%':<10}")
    print("-" * 80)
    
    for trade in trade_results[-10:]:
        print(f"{trade['buy_date']:<12} {trade['sell_date']:<12} "
              f"{trade['buy_price']:<12.2f} {trade['sell_price']:<12.2f} "
              f"{trade['pnl_percent']:<10.2f}%")
    
    # 分析连续亏损
    print("\n连续亏损分析:")
    consecutive_losses = 0
    max_consecutive_losses = 0
    current_loss_streak = 0
    
    for trade in trade_results:
        if trade['pnl_percent'] < 0:
            current_loss_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
        else:
            current_loss_streak = 0
    
    print(f"最大连续亏损次数: {max_consecutive_losses}")
    
    # 计算亏损交易的平均持有时间
    loss_durations = []
    for trade in trade_results:
        if trade['pnl_percent'] < 0:
            duration = (trade['sell_date'] - trade['buy_date']).days
            loss_durations.append(duration)
    
    if loss_durations:
        avg_loss_duration = sum(loss_durations) / len(loss_durations)
        print(f"亏损交易平均持有天数: {avg_loss_duration:.1f}")
    
    # 分析亏损交易的价格模式
    print("\n亏损交易特征分析:")
    loss_trades = [t for t in trade_results if t['pnl_percent'] < 0]
    
    if loss_trades:
        avg_loss = sum(t['pnl_percent'] for t in loss_trades) / len(loss_trades)
        max_loss = min(t['pnl_percent'] for t in loss_trades)
        print(f"平均亏损幅度: {avg_loss:.2f}%")
        print(f"最大单笔亏损: {max_loss:.2f}%")
        
        # 分析亏损交易的买入价格相对位置
        high_prices = []
        for trade in loss_trades:
            # 这里需要获取交易期间的最高价，但简化处理
            high_prices.append(trade['buy_price'])
        
        print(f"亏损交易平均买入价格: {sum(high_prices)/len(high_prices):.2f}")
    
    print("="*50)


def main():
    """
    主函数
    """
    try:
        print("开始趋势线突破回测...")
        
        # 获取脚本所在目录
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, "BTCUSDT3600.csv")
        
        # 1. 数据预处理
        print("1. 数据预处理...")
        df = preprocess_data(data_file)
        print(f"   数据形状: {df.shape}")
        
        # 2. 信号生成
        print("2. 信号生成...")
        df_with_signals = generate_signals(df)
        long_signals = df_with_signals['long_signal'].sum()
        exit_signals = df_with_signals['exit_signal'].sum()
        print(f"   做多信号数量: {long_signals}")
        print(f"   平仓信号数量: {exit_signals}")
        
        # 3. 执行回测
        print("3. 执行回测...")
        results, cerebro = run_backtest(df_with_signals)
        
        # 4. 输出结果
        print("4. 输出结果...")
        print_results(results, cerebro)
        
        # 5. 交易分析
        print("5. 交易分析...")
        analyze_trades(results, cerebro)
        
        print("回测完成！")
        
    except FileNotFoundError:
        print("错误: 找不到数据文件 'BTCUSDT3600.csv'")
    except ValueError as e:
        print(f"数据错误: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
