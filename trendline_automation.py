import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    """
    检查趋势线是否有效

    参数:
        support: 布尔值,True表示支撑线,False表示阻力线
        pivot: 整数,枢轴点的索引位置
        slope: 浮点数,趋势线的斜率
        y: numpy数组,价格数据

    返回:
        如果趋势线无效则返回-1.0,否则返回误差平方和
    """
    # 计算通过枢轴点且具有给定斜率的直线的截距
    # 使用点斜式方程: y = slope * x + intercept
    # 变换得到: intercept = y - slope * x
    intercept = -slope * pivot + y[pivot]

    # 计算趋势线上所有点的y值
    line_vals = slope * np.arange(len(y)) + intercept

    # 计算趋势线值与实际价格的差值
    # 正值表示趋势线在价格上方,负值表示趋势线在价格下方
    diffs = line_vals - y

    # 检查趋势线是否有效,如果无效则返回-1
    # 对于支撑线:所有价格点都应该在线上方或线上(diffs应该<=0)
    # 如果最大差值>0(即有价格点在支撑线下方),则该支撑线无效
    if support and diffs.max() > 1e-5:
        return -1.0
    # 对于阻力线:所有价格点都应该在线下方或线上(diffs应该>=0)
    # 如果最小差值<0(即有价格点在阻力线上方),则该阻力线无效
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # 计算数据点与趋势线之间差值的平方和,作为拟合误差
    # 误差越小说明趋势线与数据拟合得越好
    err = (diffs**2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    """
    优化趋势线的斜率,使其最佳拟合价格数据

    参数:
        support: 布尔值,True表示支撑线,False表示阻力线
        pivot: 整数,枢轴点的索引位置
        init_slope: 浮点数,初始斜率(通常使用最小二乘法得到)
        y: numpy数组,价格数据

    返回:
        元组 (最优斜率, 截距)
    """
    # 斜率变化的单位量,基于价格范围和数据长度来确定
    # 这个单位代表每个数据点对应的平均价格变化
    slope_unit = (y.max() - y.min()) / len(y)

    # 优化算法的相关变量
    opt_step = 1.0  # 初始优化步长
    min_step = 0.0001  # 最小步长阈值,当步长小于此值时停止优化
    curr_step = opt_step  # 当前步长

    # 使用初始斜率(最小二乘法的结果)作为起始点
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert best_err >= 0.0  # 初始斜率应该总是有效的,否则数据有问题

    # 梯度下降优化的标志和变量
    get_derivative = True  # 是否需要重新计算导数(梯度方向)
    derivative = None  # 导数值,用于确定优化方向

    # 迭代优化,直到步长小于最小阈值
    while curr_step > min_step:

        if get_derivative:
            # 数值微分法计算导数:
            # 将斜率增加一个很小的量,观察误差如何变化
            # 这样可以得到误差函数的梯度方向
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            # 如果增加斜率导致趋势线无效(返回负值)
            # 则尝试减小斜率来计算导数
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            # 如果两个方向都失败,说明数据有问题
            if test_err < 0.0:
                raise Exception("导数计算失败,请检查您的数据")

            get_derivative = False

        # 根据导数(梯度)决定优化方向
        if derivative > 0.0:
            # 增加斜率会增加误差,所以应该减小斜率
            test_slope = best_slope - slope_unit * curr_step
        else:
            # 增加斜率会减小误差,所以应该增加斜率
            test_slope = best_slope + slope_unit * curr_step

        # 测试新的斜率
        test_err = check_trend_line(support, pivot, test_slope, y)

        if test_err < 0 or test_err >= best_err:
            # 新斜率无效或者没有减小误差
            # 减小步长,进行更精细的搜索
            curr_step *= 0.5
        else:
            # 新斜率有效且减小了误差
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # 需要重新计算导数,因为我们移动到了新位置

    # 优化完成,返回最优斜率和对应的截距
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    """
    基于单一价格序列(如收盘价)拟合支撑线和阻力线

    参数:
        data: numpy数组,价格数据序列

    返回:
        元组 ((支撑线斜率, 支撑线截距), (阻力线斜率, 阻力线截距))
    """
    # 使用最小二乘法拟合一条基准线
    # x是数据点的索引 [0, 1, 2, ..., n-1]
    x = np.arange(len(data))
    # 进行一次多项式拟合(线性拟合)
    # coefs[0] = 斜率, coefs[1] = 截距
    coefs = np.polyfit(x, data, 1)

    # 计算基准线上所有点的值
    line_points = coefs[0] * x + coefs[1]

    # 找到上下枢轴点:
    # 上枢轴点:实际价格与基准线差值最大的点(价格最高于基准线的点)
    upper_pivot = (data - line_points).argmax()
    # 下枢轴点:实际价格与基准线差值最小的点(价格最低于基准线的点)
    lower_pivot = (data - line_points).argmin()

    # 以基准线的斜率为初始值,优化支撑线和阻力线的斜率
    # 支撑线通过下枢轴点
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    # 阻力线通过上枢轴点
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    """
    基于K线的最高价、最低价和收盘价拟合支撑线和阻力线
    这个方法更适合K线图数据,因为它同时考虑了价格的上下边界

    参数:
        high: numpy数组,最高价序列
        low: numpy数组,最低价序列
        close: numpy数组,收盘价序列

    返回:
        元组 ((支撑线斜率, 支撑线截距), (阻力线斜率, 阻力线截距))
    """
    # 基于收盘价使用最小二乘法拟合基准线
    x = np.arange(len(close))
    # coefs[0] = 斜率, coefs[1] = 截距
    coefs = np.polyfit(x, close, 1)

    # 计算基准线上所有点的值
    line_points = coefs[0] * x + coefs[1]

    # 找到枢轴点:
    # 上枢轴点:使用最高价序列,找到最高价与基准线差值最大的点
    upper_pivot = (high - line_points).argmax()
    # 下枢轴点:使用最低价序列,找到最低价与基准线差值最小的点
    lower_pivot = (low - line_points).argmin()

    # 优化趋势线:
    # 支撑线:通过下枢轴点,基于最低价序列进行优化
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    # 阻力线:通过上枢轴点,基于最高价序列进行优化
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


if __name__ == "__main__":

    # 加载CSV格式的价格数据
    data = pd.read_csv("BTCUSDT86400.csv")
    # 将日期列转换为datetime格式
    data["date"] = data["date"].astype("datetime64[s]")
    # 将日期设置为索引
    data = data.set_index("date")

    # 对价格数据取自然对数,解决价格尺度问题
    # 对数变换可以将指数增长转换为线性增长,使趋势线更加稳定
    data = np.log(data)

    # 趋势线计算的回溯周期(窗口大小)
    # 这里使用30个数据点(对于日线数据就是30天)
    lookback = 30

    # 初始化存储斜率的列表,初始值为NaN
    support_slope = [np.nan] * len(data)
    resist_slope = [np.nan] * len(data)

    # 滚动窗口计算:对每个时间点计算其对应的趋势线
    # 从第lookback-1个数据点开始(因为前面的点没有足够的历史数据)
    for i in range(lookback - 1, len(data)):
        # 提取当前窗口的K线数据(lookback个数据点)
        candles = data.iloc[i - lookback + 1 : i + 1]

        # 计算当前窗口的支撑线和阻力线系数
        support_coefs, resist_coefs = fit_trendlines_high_low(
            candles["high"], candles["low"], candles["close"]
        )
        # 只保存斜率值(索引0是斜率,索引1是截距)
        support_slope[i] = support_coefs[0]
        resist_slope[i] = resist_coefs[0]

    # 将计算出的斜率添加到数据框中
    data["support_slope"] = support_slope
    data["resist_slope"] = resist_slope

    # 绘制图表
    # 使用深色背景样式
    plt.style.use("dark_background")
    # 创建图表和主坐标轴
    fig, ax1 = plt.subplots()
    # 创建共享x轴的第二个y轴(用于显示斜率)
    ax2 = ax1.twinx()

    # 在主坐标轴上绘制收盘价
    data["close"].plot(ax=ax1)
    # 在第二坐标轴上绘制支撑线斜率(绿色)
    data["support_slope"].plot(ax=ax2, label="支撑线斜率", color="green")
    # 在第二坐标轴上绘制阻力线斜率(红色)
    data["resist_slope"].plot(ax=ax2, label="阻力线斜率", color="red")

    # 设置图表标题
    plt.title("BTC-USDT日线趋势线斜率")
    # 显示图例
    plt.legend()
    # 显示图表
    plt.show()
