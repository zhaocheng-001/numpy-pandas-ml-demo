# NumPy & Pandas + 机器学习极简 Demo

电商用户流失分析小案例：NumPy/Pandas 基础 → 数据清洗 → Matplotlib 可视化 → sklearn 逻辑回归建模评估，一条命令跑通「读取 → 清洗 → 建模 → 评估」全流程，并输出简短分析报告。

## 技术点对照

| 清单项 | 对应代码 |
| ---- | ---- |
| NumPy 数组、向量化运算、切片 | `basics/numpy_basics.py`（含 100 万元素向量化 vs 循环性能对比） |
| Pandas Series / DataFrame / 读 csv | `basics/pandas_basics.py` |
| 数据清洗：缺失值、重复值 | `analysis/data_cleaning.py`（去重 + 中位数填充 + 异常值检查） |
| Matplotlib 折线图、柱状图 | `analysis/visualize.py`（中文渲染、柱顶标数、保存 PNG） |
| sklearn 数据集划分、逻辑回归 | `analysis/ml_demo.py`（train_test_split 分层 + Pipeline 标准化） |
| 完整案例：读取-清洗-建模-评估 | `main_analysis.py` 一键跑通 |
| 简短分析报告 | `docs/分析报告.md`（基于真实运行结果撰写） |

## 项目结构

```
numpy-pandas-ml-demo/
├── basics/
│   ├── numpy_basics.py      # 数组创建、向量化、切片、布尔索引
│   └── pandas_basics.py     # Series、DataFrame、groupby、read_csv
├── analysis/
│   ├── data_cleaning.py     # 去重复、缺失值中位数填充、city 缺失删行、异常检查
│   ├── visualize.py         # 折线图（流失趋势）/ 柱状图（城市消费）/ 直方图（年龄）
│   └── ml_demo.py           # 逻辑回归：Pipeline 标准化 + 评估 + 特征系数解读
├── data/generate_data.py    # 模拟 1000 用户，故意注入 5% 缺失 + 3% 重复
├── output/                  # 生成物：3 张图表、清洗后 csv、metrics.json
├── docs/分析报告.md          # 数据分析结论（含图表）
├── tests/smoke_test.py      # 冒烟测试 5 项
├── main_analysis.py         # 全流程入口
└── requirements.txt
```

## 快速开始

```bash
cd numpy-pandas-ml-demo
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python basics/numpy_basics.py   # 1. NumPy 基础演示
python basics/pandas_basics.py  # 2. Pandas 基础演示
python main_analysis.py         # 3. 一键完整案例（推荐）
python tests/smoke_test.py      # 4. 冒烟测试
```

## 一键案例的 5 个步骤

`python main_analysis.py` 依次执行：

1. **生成数据**：模拟 1030 行用户数据，故意注入 156 个缺失值、30 条重复行
2. **清洗**：去重 30 条、数值列中位数填充、城市缺失删行 → 干净的 1000 行
3. **可视化**：流失率随在网时长折线图、各城市消费柱状图、年龄分布直方图
4. **建模**：逻辑回归预测用户流失（80/20 划分、标准化 Pipeline）
5. **评估**：测试集准确率 **78.5%**（precision/recall 均衡），导出特征系数与指标

核心结论：**在网时长和月消费是流失的最强影响因素**——新用户前 6 个月流失率高达 75%+，是留存运营的关键窗口。详见 `docs/分析报告.md`。

## 数据说明

数据由 `data/generate_data.py` 用固定随机种子合成：流失概率真实依赖「消费金额 + 在网时长 + 年龄」（sigmoid 关系），因此模型能学到可解释的规律；同时注入脏数据模拟真实场景。替换成真实业务数据后，同一套代码可直接复用。
