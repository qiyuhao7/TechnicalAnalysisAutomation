import pandas as pd
import numpy as np
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
