# 趋势线突破回测设计文档

## 1. 概述

基于现有的 `TrendlineOptimizer` 类，创建一个 backtrader 回测脚本，用于测试趋势线突破策略。策略逻辑：当价格向上突破下降趋势线（阻力线）时做多，当价格跌破上升趋势线（支撑线）时平仓。

## 2. 需求总结

- **数据源**：`BTCUSDT3600.csv`（BTC/USDT 1小时K线数据）
- **策略类型**：趋势线突破策略（仅做多）
- **进场条件**：价格向上突破下降趋势线（阻力线），突破阈值为 ATR(14) 的 0.5 倍
- **出场条件**：价格跌破上升趋势线（支撑线），跌破阈值为 ATR(14) 的 0.5 倍
- **资金管理**：每次交易使用全部可用资金
- **交易成本**：无手续费、无滑点
- **回测指标**：收益率、最大回撤、胜率等基础指标
- **时间范围**：整个数据集（2018年至今）

## 3. 架构设计

### 3.1 整体流程

```
输入数据 (BTCUSDT3600.csv)
    ↓
预处理阶段 (TrendlineOptimizer)
    ↓
信号生成阶段 (基于趋势线特征)
    ↓
回测阶段 (backtrader)
    ↓
结果输出 (绩效报告)
```

### 3.2 数据流

1. **输入数据**：`BTCUSDT3600.csv`
   - 列：date, close, open, high, low
   - 格式：CSV，日期索引

2. **预处理输出**：添加趋势线特征列
   - `support_slope_logspace`：支撑线斜率（对数空间）
   - `resist_slope_logspace`：阻力线斜率（对数空间）
   - `support_pivot_val_real`：支撑线枢轴点实际值
   - `resist_pivot_val_real`：阻力线枢轴点实际值
   - `support_pivot_abs`：支撑线枢轴点绝对索引
   - `resist_pivot_abs`：阻力线枢轴点绝对索引

3. **信号生成**：基于趋势线特征生成交易信号
   - `long_signal`：做多信号（1表示进场，0表示无信号）
   - `exit_signal`：平仓信号（1表示平仓，0表示无信号）

4. **backtrader 输入**：将数据和信号转换为 backtrader 可接受的格式

## 4. 交易逻辑

### 4.1 做多信号生成

**条件**：
1. 当前收盘价向上突破下降趋势线（阻力线）
2. 突破幅度超过 ATR(14) 的 0.5 倍

**计算方法**：
```python
# 计算突破阈值
atr = ATR(high, low, close, period=14)
breakout_threshold = atr * 0.5

# 检查突破条件
resistance_line_value = resist_pivot_val_real + resist_slope_logspace * (当前索引 - resist_pivot_abs)
if close > resistance_line_value + breakout_threshold:
    long_signal = 1
```

### 4.2 平仓信号生成

**条件**：
1. 当前收盘价跌破上升趋势线（支撑线）
2. 跌破幅度超过 ATR(14) 的 0.5 倍

**计算方法**：
```python
# 计算跌破阈值
breakdown_threshold = atr * 0.5

# 检查跌破条件
support_line_value = support_pivot_val_real + support_slope_logspace * (当前索引 - support_pivot_abs)
if close < support_line_value - breakdown_threshold:
    exit_signal = 1
```

## 5. backtrader 策略实现

### 5.1 策略类

```python
class TrendlineBreakoutStrategy(bt.Strategy):
    def __init__(self):
        self.long_signal = self.data.long_signal
        self.exit_signal = self.data.exit_signal
        
    def next(self):
        if self.long_signal[0] == 1 and not self.position:
            self.buy(size=self.broker.getcash() / self.data.close[0])
        elif self.exit_signal[0] == 1 and self.position:
            self.sell(size=self.position.size)
```

### 5.2 数据加载

```python
data = bt.feeds.PandasData(
    dataname=preprocessed_df,
    datetime=None,
    open='open',
    high='high',
    low='low',
    close='close',
    volume=None,
    openinterest=-1
)
```

### 5.3 绩效指标

使用 backtrader 内置分析器：
- `bt.analyzers.Returns`：计算收益率
- `bt.analyzers.DrawDown`：计算最大回撤
- `bt.analyzers.SharpeRatio`：计算夏普比率
- `bt.analyzers.TradeAnalyzer`：计算交易统计

## 6. 文件结构

```
trendline_backtest.py          # 主脚本
├── 预处理函数                  # 使用 TrendlineOptimizer 处理数据
├── 信号生成函数                # 基于趋势线特征生成交易信号
├── backtrader 策略类           # 实现交易逻辑
├── 回测执行函数                # 配置和运行回测
└── 结果输出函数                # 打印绩效报告
```

## 7. 错误处理

1. **数据验证**：
   - 检查输入文件是否存在
   - 检查必需列是否存在（date, close, open, high, low）
   - 检查数据类型和缺失值

2. **参数验证**：
   - ATR 周期必须为正整数
   - 突破阈值必须为正数

3. **边界情况**：
   - 数据量不足时抛出异常
   - 趋势线计算失败时跳过该时间点

4. **异常处理**：
   - 捕获文件读取异常
   - 捕获计算异常
   - 提供有意义的错误信息

## 8. 使用方式

```bash
python trendline_backtest.py
```

输出：
- 回测绩效报告（收益率、最大回撤、胜率等）
- 交易统计（交易次数、胜率等）

## 9. 依赖项

- `pandas`
- `numpy`
- `backtrader`
- `trendline_automation.py`（现有模块）

## 10. 未来扩展

1. **参数优化**：支持对 ATR 周期、突破阈值等参数进行优化
2. **可视化**：生成交易信号图表和权益曲线
3. **多品种支持**：支持多个交易品种的回测
4. **止损止盈**：添加固定止损止盈功能