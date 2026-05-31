# 信号生成函数设计文档

## 概述

为趋势线突破回测系统添加信号生成功能，基于ATR（平均真实波幅）计算突破阈值，生成做多和平仓交易信号。

## 组件设计

### 1. ATR计算函数

**函数签名：**
```python
def calculate_atr(high, low, close, period=14)
```

**功能：**
- 计算真实波幅均值（ATR）
- 使用标准ATR计算方法：TR = max(high-low, |high-prev_close|, |low-prev_close|)
- 返回ATR的移动平均序列

**参数：**
- `high`: 最高价序列
- `low`: 最低价序列  
- `close`: 收盘价序列
- `period`: ATR周期（默认14）

### 2. 信号生成函数

**函数签名：**
```python
def generate_signals(df, atr_period=14, atr_multiplier=0.5)
```

**功能：**
- 基于趋势线特征生成交易信号
- 计算ATR和突破阈值
- 生成做多信号（突破阻力线）
- 生成平仓信号（跌破支撑线）

**信号逻辑：**
1. **做多信号**：当收盘价 > 阻力线值 + 突破阈值时
2. **平仓信号**：当收盘价 < 支撑线值 - 突破阈值时

**趋势线值计算：**
- 使用对数空间计算：`log_value = log(pivot_val_real) + slope * (i - pivot_abs)`
- 转换回实际值：`value = exp(log_value)`

### 3. 测试代码

**测试流程：**
1. 使用`preprocess_data`加载数据
2. 调用`generate_signals`生成信号
3. 统计并输出信号数量

## 数据流

```
原始数据 → preprocess_data → TrendlineOptimizer → 包含趋势线特征的DataFrame
                ↓
        generate_signals → 包含交易信号的DataFrame
```

## 依赖关系

- `trendline_automation.py`: TrendlineOptimizer类
- `pandas`, `numpy`: 数据处理
- `BTCUSDT3600.csv`: 测试数据

## 验收标准

1. ATR计算函数正确实现标准ATR算法
2. 信号生成函数能正确识别突破和跌破
3. 测试代码能成功运行并输出信号统计
4. 函数修改原始DataFrame并返回