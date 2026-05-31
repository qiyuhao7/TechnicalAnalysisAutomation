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


if __name__ == "__main__":
    try:
        df = preprocess_data("BTCUSDT3600.csv")
        print(f"数据预处理成功，数据形状: {df.shape}")
        print(f"趋势线特征列: {[col for col in df.columns if 'slope' in col or 'pivot' in col]}")
    except Exception as e:
        print(f"数据预处理失败: {e}")
