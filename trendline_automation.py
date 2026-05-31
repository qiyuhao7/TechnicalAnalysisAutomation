import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray, x: np.ndarray, valid_mask: np.ndarray):
    """
    检查趋势线有效性（带软约束屏蔽）
    验证时只检查 valid_mask 为 True 的点，忽略被分位数剔除的极端异常点。
    """
    intercept = -slope * pivot + y[pivot]
    diffs = (slope * x + intercept) - y

    # 在合法数据点（非极端插针点）中检查是否穿透
    if support and diffs[valid_mask].max() > 1e-5:
        return -1.0
    elif not support and diffs[valid_mask].min() < -1e-5:
        return -1.0

    # 仅对合法数据点计算误差平方和，防止极端点拉大误差导致斜率失真
    err = (diffs[valid_mask]**2.0).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray, x: np.ndarray, valid_mask: np.ndarray):
    """
    带掩码约束的梯度下降斜率优化
    """
    slope_unit = (y.max() - y.min()) / len(y)
    if slope_unit == 0: 
        return init_slope, -init_slope * pivot + y[pivot]

    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step
    best_slope = init_slope

    best_err = check_trend_line(support, pivot, init_slope, y, x, valid_mask)
    
    # 如果初始基准线由于浮点精度已经判定无效，做一次容错保护
    if best_err < 0:
        return init_slope, -init_slope * pivot + y[pivot]

    get_derivative = True
    derivative = None

    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y, x, valid_mask)
            
            if test_err >= 0.0:
                derivative = test_err - best_err
            else:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y, x, valid_mask)
                if test_err >= 0.0:
                    derivative = best_err - test_err
                else:
                    break # 两侧均无法求导（处于极窄的有效夹角内），停止优化

            get_derivative = False

        if derivative is not None and derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y, x, valid_mask)

        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5  # 撞墙或误差变大，步长减半
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True 

    return (best_slope, -best_slope * pivot + y[pivot])


if __name__ == "__main__":
    # ==========================
    # 1. 数据加载与预处理
    # ==========================
    try:
        data = pd.read_csv("BTCUSDT86400.csv")
    except FileNotFoundError:
        # 生成模拟数据以供测试运行
        print("未找到数据文件，使用正弦波叠加随机噪音的模拟数据。")
        np.random.seed(42)
        sim_len = 200
        base = np.linspace(10000, 20000, sim_len) + np.sin(np.linspace(0, 10, sim_len)) * 2000
        data = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=sim_len),
            "open": base + np.random.normal(0, 300, sim_len),
            "close": base + np.random.normal(0, 300, sim_len)
        })
        data["high"] = data[["open", "close"]].max(axis=1) + np.random.exponential(500, sim_len)
        data["low"] = data[["open", "close"]].min(axis=1) - np.random.exponential(500, sim_len)
        
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    # 对数变换（处理指数级价格）
    cols_to_log = ['open', 'high', 'low', 'close']
    data[cols_to_log] = np.log(data[cols_to_log])

    # ==========================
    # 2. 核心优化 1：使用K线实体替代影线
    # ==========================
    opens = data['open'].values
    closes = data['close'].values
    
    # 支撑线使用 min(开盘,收盘)，阻力线使用 max(开盘,收盘)
    max_body = np.maximum(opens, closes)
    min_body = np.minimum(opens, closes)

    lookback = 30
    support_slope = np.full(len(data), np.nan)
    resist_slope = np.full(len(data), np.nan)
    
    # 性能优化：提前生成静态 X 轴数组
    x_window = np.arange(lookback)

    # ==========================
    # 3. 向量化滚动窗口计算
    # ==========================
    for i in range(lookback - 1, len(data)):
        w_min = min_body[i - lookback + 1 : i + 1]
        w_max = max_body[i - lookback + 1 : i + 1]
        w_close = closes[i - lookback + 1 : i + 1]

        # 拟合基准线
        coefs = np.polyfit(x_window, w_close, 1)
        line_points = coefs[0] * x_window + coefs[1]

        # ==========================
        # 核心优化 3：5% 分位数过滤
        # ==========================
        # -- 支撑线枢轴点寻找 --
        diffs_lower = w_min - line_points
        threshold_lower = np.percentile(diffs_lower, 5) # 找到底部 5% 的阈值
        valid_lower_mask = diffs_lower >= threshold_lower # 过滤掉跌破 5% 阈值的极端异常实体
        # 在合法数据中寻找最低点作为枢轴点
        lower_pivot = np.where(valid_lower_mask, diffs_lower, np.inf).argmin()

        # -- 阻力线枢轴点寻找 --
        diffs_upper = w_max - line_points
        threshold_upper = np.percentile(diffs_upper, 95) # 找到顶部 5% 的阈值
        valid_upper_mask = diffs_upper <= threshold_upper
        upper_pivot = np.where(valid_upper_mask, diffs_upper, -np.inf).argmax()

        # 优化趋势线斜率（传入 valid_mask）
        s_slope, s_int = optimize_slope(True, lower_pivot, coefs[0], w_min, x_window, valid_lower_mask)
        r_slope, r_int = optimize_slope(False, upper_pivot, coefs[0], w_max, x_window, valid_upper_mask)

        support_slope[i] = s_slope
        resist_slope[i] = r_slope

    data["support_slope"] = support_slope
    data["resist_slope"] = resist_slope

    # ==========================
    # 4. 绘图展示
    # ==========================
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 画回真实的收盘价（为了直观展示，这里将对数还原）
    np.exp(data["close"]).plot(ax=ax1, color='white', alpha=0.8, label='Close (Real Price)')
    
    # 绘制斜率（斜率本身处于对数空间，变化平缓）
    data["support_slope"].plot(ax=ax2, label="Support Line Slope", color="#00ff00", linewidth=2)
    # 因为你只做多，阻力线斜率设为透明度较低的辅助线
    data["resist_slope"].plot(ax=ax2, label="Resistance Line Slope", color="#ff0000", linewidth=1, alpha=0.4)

    plt.title("BTC-USDT Optimized Trendline Slopes (Long-Only Strategy)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()