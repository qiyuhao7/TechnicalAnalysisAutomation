# 趋势线突破回测实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于现有的 TrendlineOptimizer 类，创建一个 backtrader 回测脚本，用于测试趋势线突破策略。

**Architecture:** 采用预计算方法，先使用 TrendlineOptimizer 处理数据并生成交易信号，然后使用 backtrader 框架执行回测。策略逻辑：当价格向上突破下降趋势线（阻力线）时做多，当价格跌破上升趋势线（支撑线）时平仓。

**Tech Stack:** Python, pandas, numpy, backtrader, trendline_automation.py

---

## 文件结构

- **创建**: `trendline_backtest.py` - 主脚本，包含预处理、信号生成、backtrader 回测和结果输出
- **修改**: 无（直接使用现有的 `trendline_automation.py`）
- **测试**: 无（基础回测，不包含单元测试）

## 任务分解

### Task 1: 数据预处理函数

**Files:**
- Create: `trendline_backtest.py`
- Import: `trendline_automation.py`

- [ ] **Step 1: 创建数据预处理函数**

```python
import pandas as pd
import numpy as np
from trendline_automation import TrendlineOptimizer

def preprocess_data(file_path, lookback=30):
    """
    预处理数据，使用 TrendlineOptimizer 计算趋势线特征
    
    Args:
        file_path: CSV 文件路径
        lookback: TrendlineOptimizer 的 lookback 参数
        
    Returns:
        包含趋势线特征的 DataFrame
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 检查必需列
    required_cols = ['date', 'close', 'open', 'high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入数据缺失必需列: '{col}'")
    
    # 设置日期索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 使用 TrendlineOptimizer 处理数据
    optimizer = TrendlineOptimizer(lookback=lookback)
    df_with_features = optimizer.fit_transform(df)
    
    return df_with_features
```

- [ ] **Step 2: 测试数据预处理函数**

```python
# 测试代码
if __name__ == "__main__":
    try:
        df = preprocess_data("BTCUSDT3600.csv")
        print(f"数据预处理成功，数据形状: {df.shape}")
        print(f"趋势线特征列: {[col for col in df.columns if 'slope' in col or 'pivot' in col]}")
    except Exception as e:
        print(f"数据预处理失败: {e}")
```

- [ ] **Step 3: 运行测试验证**

Run: `python trendline_backtest.py`
Expected: 输出数据预处理成功信息和趋势线特征列

### Task 2: 信号生成函数

**Files:**
- Modify: `trendline_backtest.py`

- [ ] **Step 1: 创建 ATR 计算函数**

```python
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
```

- [ ] **Step 2: 创建信号生成函数**

```python
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
    # 计算 ATR
    atr = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    
    # 计算突破阈值
    breakout_threshold = atr * atr_multiplier
    
    # 初始化信号列
    df['long_signal'] = 0
    df['exit_signal'] = 0
    
    # 生成做多信号
    for i in range(1, len(df)):
        # 检查是否有有效的阻力线数据
        if pd.isna(df['resist_pivot_val_real'].iloc[i]) or pd.isna(df['resist_slope_logspace'].iloc[i]):
            continue
            
        # 计算当前阻力线值
        resist_pivot_abs = df['resist_pivot_abs'].iloc[i]
        if resist_pivot_abs < 0:
            continue
            
        # 计算阻力线值（对数空间）
        resist_line_value_log = np.log(df['resist_pivot_val_real'].iloc[i]) + \
                               df['resist_slope_logspace'].iloc[i] * (i - resist_pivot_abs)
        resist_line_value = np.exp(resist_line_value_log)
        
        # 检查突破条件
        current_close = df['close'].iloc[i]
        threshold = breakout_threshold.iloc[i] if not pd.isna(breakout_threshold.iloc[i]) else 0
        
        if current_close > resist_line_value + threshold:
            df.iloc[i, df.columns.get_loc('long_signal')] = 1
    
    # 生成平仓信号
    for i in range(1, len(df)):
        # 检查是否有有效的支撑线数据
        if pd.isna(df['support_pivot_val_real'].iloc[i]) or pd.isna(df['support_slope_logspace'].iloc[i]):
            continue
            
        # 计算当前支撑线值
        support_pivot_abs = df['support_pivot_abs'].iloc[i]
        if support_pivot_abs < 0:
            continue
            
        # 计算支撑线值（对数空间）
        support_line_value_log = np.log(df['support_pivot_val_real'].iloc[i]) + \
                                df['support_slope_logspace'].iloc[i] * (i - support_pivot_abs)
        support_line_value = np.exp(support_line_value_log)
        
        # 检查跌破条件
        current_close = df['close'].iloc[i]
        threshold = breakout_threshold.iloc[i] if not pd.isna(breakout_threshold.iloc[i]) else 0
        
        if current_close < support_line_value - threshold:
            df.iloc[i, df.columns.get_loc('exit_signal')] = 1
    
    return df
```

- [ ] **Step 3: 测试信号生成函数**

```python
# 测试代码
if __name__ == "__main__":
    try:
        df = preprocess_data("BTCUSDT3600.csv")
        df_with_signals = generate_signals(df)
        
        long_signals = df_with_signals['long_signal'].sum()
        exit_signals = df_with_signals['exit_signal'].sum()
        
        print(f"信号生成成功")
        print(f"做多信号数量: {long_signals}")
        print(f"平仓信号数量: {exit_signals}")
    except Exception as e:
        print(f"信号生成失败: {e}")
```

- [ ] **Step 4: 运行测试验证**

Run: `python trendline_backtest.py`
Expected: 输出信号生成成功信息和信号数量

### Task 3: backtrader 策略类

**Files:**
- Modify: `trendline_backtest.py`

- [ ] **Step 1: 创建 backtrader 策略类**

```python
import backtrader as bt

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
```

- [ ] **Step 2: 测试策略类**

```python
# 测试代码
if __name__ == "__main__":
    try:
        # 测试策略类是否可以正确实例化
        strategy = TrendlineBreakoutStrategy()
        print("策略类创建成功")
    except Exception as e:
        print(f"策略类创建失败: {e}")
```

- [ ] **Step 3: 运行测试验证**

Run: `python trendline_backtest.py`
Expected: 输出策略类创建成功信息

### Task 4: 回测执行函数

**Files:**
- Modify: `trendline_backtest.py`

- [ ] **Step 1: 创建回测执行函数**

```python
def run_backtest(df, initial_cash=10000):
    """
    执行回测
    
    Args:
        df: 包含交易信号的 DataFrame
        initial_cash: 初始资金
        
    Returns:
        回测结果
    """
    # 创建 cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(TrendlineBreakoutStrategy)
    
    # 准备数据
    # 添加额外列供 backtrader 使用
    df_bt = df[['open', 'high', 'low', 'close', 'long_signal', 'exit_signal']].copy()
    
    # 创建数据源
    data = bt.feeds.PandasData(
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
```

- [ ] **Step 2: 测试回测执行函数**

```python
# 测试代码
if __name__ == "__main__":
    try:
        df = preprocess_data("BTCUSDT3600.csv")
        df_with_signals = generate_signals(df)
        
        results, cerebro = run_backtest(df_with_signals)
        print("回测执行成功")
    except Exception as e:
        print(f"回测执行失败: {e}")
```

- [ ] **Step 3: 运行测试验证**

Run: `python trendline_backtest.py`
Expected: 输出回测执行成功信息

### Task 5: 结果输出函数

**Files:**
- Modify: `trendline_backtest.py`

- [ ] **Step 1: 创建结果输出函数**

```python
def print_results(results, cerebro):
    """
    打印回测结果
    
    Args:
        results: 回测结果
        cerebro: cerebro 引擎
    """
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
```

- [ ] **Step 2: 测试结果输出函数**

```python
# 测试代码
if __name__ == "__main__":
    try:
        df = preprocess_data("BTCUSDT3600.csv")
        df_with_signals = generate_signals(df)
        
        results, cerebro = run_backtest(df_with_signals)
        print_results(results, cerebro)
        print("结果输出成功")
    except Exception as e:
        print(f"结果输出失败: {e}")
```

- [ ] **Step 3: 运行测试验证**

Run: `python trendline_backtest.py`
Expected: 输出完整的回测结果报告

### Task 6: 主函数和错误处理

**Files:**
- Modify: `trendline_backtest.py`

- [ ] **Step 1: 创建主函数**

```python
def main():
    """
    主函数
    """
    try:
        print("开始趋势线突破回测...")
        
        # 1. 数据预处理
        print("1. 数据预处理...")
        df = preprocess_data("BTCUSDT3600.csv")
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
        
        print("回测完成！")
        
    except FileNotFoundError:
        print("错误: 找不到数据文件 'BTCUSDT3600.csv'")
    except ValueError as e:
        print(f"数据错误: {e}")
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
```

- [ ] **Step 2: 更新主程序入口**

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行完整回测**

Run: `python trendline_backtest.py`
Expected: 输出完整的回测流程和结果报告

- [ ] **Step 4: 提交代码**

```bash
git add trendline_backtest.py
git commit -m "feat: 添加趋势线突破回测脚本"
```

## 自检清单

1. **规范覆盖**：所有设计文档中的需求都已在任务中实现
2. **占位符检查**：没有发现 TBD、TODO 或不完整的部分
3. **类型一致性**：函数名、变量名、参数名保持一致

## 执行选项

**计划完成并保存到 `docs/superpowers/plans/2026-05-31-trendline-backtest-plan.md`。两种执行选项：**

**1. Subagent-Driven (推荐)** - 我为每个任务分派一个新的子代理，任务之间进行审查，快速迭代

**2. Inline Execution** - 在当前会话中执行任务，批量执行并设置检查点

**选择哪种方式？**