import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib
import matplotlib.pyplot as plt
from trendline_automation import TrendlineOptimizer

# 设置 matplotlib 后端为非交互模式
matplotlib.use('Agg')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


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
        
        # 记录交易信息
        self.trade_records = []
        self.buy_prices = []
        self.buy_dates = []
        self.sell_prices = []
        self.sell_dates = []
        
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
            
        # 如果有持仓且出现平仓信号，则卖出
        elif self.exit_signal[0] == 1 and self.position:
            price = self.data.close[0]
            self.sell(size=self.position.size)
            
            # 记录卖出信息
            self.sell_prices.append(price)
            self.sell_dates.append(self.data.datetime.date(0))


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


def plot_equity_curve(results, cerebro, df):
    """
    绘制权益曲线和回撤图，并标注买卖点
    
    Args:
        results: 回测结果
        cerebro: cerebro 引擎
        df: 包含价格数据的 DataFrame
    """
    strategy = results[0]
    
    # 获取分析结果
    returns_analysis = strategy.analyzers.returns.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    
    # 获取交易记录
    buy_dates = strategy.buy_dates
    buy_prices = strategy.buy_prices
    sell_dates = strategy.sell_dates
    sell_prices = strategy.sell_prices
    
    # 获取实际的价格数据
    dates = df.index
    close_prices = df['close'].values
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Trendline Breakout Strategy Backtest Results', fontsize=14, fontweight='bold')
    
    # 绘制价格曲线
    ax1.plot(dates, close_prices, color='blue', linewidth=1, alpha=0.7, label='Close Price')
    
    # 标注买点
    for i, (buy_date, buy_price) in enumerate(zip(buy_dates, buy_prices)):
        ax1.scatter(buy_date, buy_price, color='green', marker='^', s=100, zorder=5)
        if i == 0:
            ax1.annotate('Buy', xy=(buy_date, buy_price), xytext=(10, 10),
                        textcoords='offset points', fontsize=8, color='green',
                        arrowprops=dict(arrowstyle='->', color='green'))
    
    # 标注卖点
    for i, (sell_date, sell_price) in enumerate(zip(sell_dates, sell_prices)):
        ax1.scatter(sell_date, sell_price, color='red', marker='v', s=100, zorder=5)
        if i == 0:
            ax1.annotate('Sell', xy=(sell_date, sell_price), xytext=(10, -15),
                        textcoords='offset points', fontsize=8, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    ax1.set_ylabel('Price (USDT)')
    ax1.set_title('Price Chart with Trade Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 计算并绘制回撤曲线
    # 使用价格数据计算回撤
    peak = np.maximum.accumulate(close_prices)
    drawdown = (peak - close_prices) / peak * 100
    
    ax2.fill_between(dates, drawdown, alpha=0.5, color='red')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Drawdown Curve')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0)
    total_return = returns_analysis.get('rtot', 0) * 100
    total_trades = len(buy_dates)
    
    textstr = f'Total Return: {total_return:.2f}%\nMax Drawdown: {max_dd:.2f}%\nTotal Trades: {total_trades}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 旋转日期标签
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(script_dir, 'backtest_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {plot_file}")
    
    # 关闭图表以释放内存
    plt.close(fig)


def plot_max_drawdown_period(results, cerebro, df):
    """
    绘制最大回撤期间的详细图表
    
    Args:
        results: 回测结果
        cerebro: cerebro 引擎
        df: 包含价格数据的 DataFrame
    """
    strategy = results[0]
    
    # 获取分析结果
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    
    # 获取交易记录
    buy_dates = strategy.buy_dates
    buy_prices = strategy.buy_prices
    sell_dates = strategy.sell_dates
    sell_prices = strategy.sell_prices
    
    # 获取实际的价格数据
    dates = df.index
    close_prices = df['close'].values
    
    # 计算回撤
    peak = np.maximum.accumulate(close_prices)
    drawdown = (peak - close_prices) / peak * 100
    
    # 找到最大回撤的位置
    max_dd_idx = np.argmax(drawdown)
    max_dd = drawdown[max_dd_idx]
    max_dd_date = dates[max_dd_idx]
    max_dd_price = close_prices[max_dd_idx]
    
    # 确定最大回撤期间（从峰值到谷值）
    # 找到峰值位置
    peak_idx = np.argmax(close_prices[:max_dd_idx + 1])
    
    # 扩大显示范围（前后各20%的数据）
    total_points = len(dates)
    margin = int(total_points * 0.1)
    start_idx = max(0, peak_idx - margin)
    end_idx = min(total_points, max_dd_idx + margin)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Maximum Drawdown Period Analysis\nMax Drawdown: {max_dd:.2f}%', 
                fontsize=14, fontweight='bold')
    
    # 绘制价格曲线（聚焦于最大回撤期间）
    period_dates = dates[start_idx:end_idx]
    period_prices = close_prices[start_idx:end_idx]
    
    ax1.plot(period_dates, period_prices, color='blue', linewidth=2, label='Close Price')
    
    # 标注峰值和谷值
    ax1.scatter(dates[peak_idx], close_prices[peak_idx], color='green', marker='^', s=200, zorder=5, label='Peak')
    ax1.scatter(dates[max_dd_idx], close_prices[max_dd_idx], color='red', marker='v', s=200, zorder=5, label='Trough')
    
    # 添加标注
    ax1.annotate(f'Peak\n{close_prices[peak_idx]:.2f}', 
                xy=(dates[peak_idx], close_prices[peak_idx]),
                xytext=(10, 20), textcoords='offset points',
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax1.annotate(f'Trough\n{close_prices[max_dd_idx]:.2f}\nDrawdown: {max_dd:.2f}%', 
                xy=(dates[max_dd_idx], close_prices[max_dd_idx]),
                xytext=(10, -30), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 标注期间的买卖点
    for buy_date, buy_price in zip(buy_dates, buy_prices):
        if start_idx <= np.searchsorted(dates, buy_date) < end_idx:
            ax1.scatter(buy_date, buy_price, color='green', marker='^', s=100, zorder=5, alpha=0.7)
    
    for sell_date, sell_price in zip(sell_dates, sell_prices):
        if start_idx <= np.searchsorted(dates, sell_date) < end_idx:
            ax1.scatter(sell_date, sell_price, color='red', marker='v', s=100, zorder=5, alpha=0.7)
    
    ax1.set_ylabel('Price (USDT)')
    ax1.set_title('Price Chart During Maximum Drawdown Period')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 绘制回撤曲线（聚焦于最大回撤期间）
    period_drawdown = drawdown[start_idx:end_idx]
    ax2.fill_between(period_dates, period_drawdown, alpha=0.5, color='red')
    ax2.scatter(dates[max_dd_idx], max_dd, color='darkred', marker='o', s=100, zorder=5)
    ax2.annotate(f'Max Drawdown: {max_dd:.2f}%', 
                xy=(dates[max_dd_idx], max_dd),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='darkred', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Drawdown Curve During Maximum Drawdown Period')
    ax2.grid(True, alpha=0.3)
    
    # 旋转日期标签
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(script_dir, 'max_drawdown_period.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nMax drawdown chart saved to: {plot_file}")
    
    # 关闭图表以释放内存
    plt.close(fig)


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
        
        # 5. 绘制图表
        print("5. 绘制图表...")
        plot_equity_curve(results, cerebro, df_with_signals)
        
        # 6. 绘制最大回撤期间图表
        print("6. 绘制最大回撤期间图表...")
        plot_max_drawdown_period(results, cerebro, df_with_signals)
        
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
