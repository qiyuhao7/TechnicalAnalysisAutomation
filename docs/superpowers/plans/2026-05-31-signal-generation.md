# 信号生成函数实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为趋势线突破回测系统添加信号生成功能，基于ATR计算突破阈值，生成做多和平仓交易信号。

**Architecture:** 在现有`trendline_backtest.py`文件中添加两个函数：ATR计算函数和信号生成函数。信号生成函数使用对数空间计算趋势线值，基于突破阈值生成交易信号。

**Tech Stack:** Python, pandas, numpy

---

## 文件结构

**修改文件：**
- `trendline_backtest.py` - 添加ATR计算函数和信号生成函数

**依赖文件：**
- `trendline_automation.py` - TrendlineOptimizer类（已存在）
- `BTCUSDT3600.csv` - 测试数据（已存在）

---

### Task 1: 添加ATR计算函数

**Files:**
- Modify: `trendline_backtest.py:1-39`

- [ ] **Step 1: 添加ATR计算函数**

在`preprocess_data`函数之前添加ATR计算函数：

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

- [ ] **Step 2: 验证函数添加成功**

Run: `python -c "import trendline_backtest; print('ATR函数添加成功')"`
Expected: 输出"ATR函数添加成功"

---

### Task 2: 添加信号生成函数

**Files:**
- Modify: `trendline_backtest.py:31-39`

- [ ] **Step 1: 添加信号生成函数**

在`preprocess_data`函数之后，`if __name__ == "__main__":`之前添加信号生成函数：

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

- [ ] **Step 2: 验证函数添加成功**

Run: `python -c "import trendline_backtest; print('信号生成函数添加成功')"`
Expected: 输出"信号生成函数添加成功"

---

### Task 3: 更新测试代码

**Files:**
- Modify: `trendline_backtest.py:33-39`

- [ ] **Step 1: 更新测试代码**

将现有的测试代码替换为包含信号生成的测试：

```python
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

- [ ] **Step 2: 运行完整测试**

Run: `python trendline_backtest.py`
Expected: 输出信号生成成功信息和信号数量

---

### Task 4: 最终验证

**Files:**
- Modify: `trendline_backtest.py`

- [ ] **Step 1: 验证代码语法**

Run: `python -m py_compile trendline_backtest.py`
Expected: 无语法错误

- [ ] **Step 2: 运行完整测试**

Run: `python trendline_backtest.py`
Expected: 输出类似以下内容：
```
信号生成成功
做多信号数量: X
平仓信号数量: Y
```

- [ ] **Step 3: 提交代码**

```bash
git add trendline_backtest.py
git commit -m "feat: add signal generation functions for trendline breakout backtest"
```