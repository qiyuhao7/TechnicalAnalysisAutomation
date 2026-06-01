# AGENTS.md — TechnicalAnalysisAutomation

## 语言

用中文回答

## 项目概览

技术形态自动化识别模块，包含趋势线、头肩顶、旗形等技术形态识别算法。使用 backtrader 进行回测。

## 知识库

Obsidian 知识库位置：`C:\Users\qiyuh\Documents\jianguoyun\Notes\政经\量化\TechnicalAnalysisAutomation`

## 运行环境

使用上一层级的 uv 环境运行 Python 脚本：

```bash
# 在 quantor/ 目录下运行 TechnicalAnalysisAutomation 脚本
uv run python TechnicalAnalysisAutomation/trendline_backtest.py
```

**重要规则：** 运行任何 Python 脚本前，必须先激活上一层级的 uv 环境：
1. 确保当前工作目录为 `quantor/`
2. 使用 `uv run python` 命令运行脚本
3. 不要使用系统 Python 或其他 Python 环境

## 关键陷阱

- **本地 import**：`TechnicalAnalysisAutomation/` 下的脚本互相 import（无包结构），运行时 cwd 必须是该目录，否则 import 失败
- **numpy 版本冲突**：需要 `numpy==1.23.1`，否则 pyclustering 不工作
- **数据文件**：`BTCUSDT3600.csv` 等数据文件在当前目录下

## 开发命令

```bash
# 运行趋势线突破回测
uv run python TechnicalAnalysisAutomation/trendline_backtest.py

# 运行测试（如果存在）
uv run python TechnicalAnalysisAutomation/test_hs_patterns.py
uv run python TechnicalAnalysisAutomation/test_flag_patterns.py
```

## 文件说明

- `trendline_backtest.py`：趋势线突破回测脚本（含 ATR 止损策略）
- `trendline_automation.py`：趋势线优化器类
- `BTCUSDT3600.csv`：BTC/USDT 1小时K线数据
- `docs/strategy_optimization_notes.md`：策略优化笔记和回测结果
- `docs/superpowers/`：设计文档和实现计划