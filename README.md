# Calcium Waveform Clustering Project

这个项目用于整理、预处理、可视化并聚类 0-20 秒钙成像波形数据。当前实验设定为刺激从 5 秒开始、10 秒结束，因此分析时间段定义为：

- 刺激前：0-5 秒
- 刺激期：5-10 秒
- 撤光后：10-20 秒

## 项目结构

```text
Calcium_Waveform_Clustering_Project/
├── data/
│   ├── included_excel/          # 已筛选纳入的 0-20 秒原始 Excel
│   ├── metadata/                # 筛选清单和整理说明
│   └── processed_tables/        # 预留：中间表格
├── notebooks/                   # 可逐格运行的 Jupyter notebook
├── scripts/                     # 可复现脚本
├── results/
│   ├── visualization/           # 原始波形可视化结果
│   ├── preprocessing/           # 预处理、峰谷、特征提取结果
│   └── clustering/              # 聚类分析结果
├── docs/                        # 项目说明和分析解释文档
├── reference/                   # 前人脚本备份
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

如果使用 Anaconda，也可以在已有环境中直接安装这些包。

## 从头运行

当前项目包已经包含整理好的纳入数据和结果。如果需要从纳入的 Excel 重新跑一遍完整流程，在项目根目录执行：

```bash
bash scripts/run_all.sh
```

等价于依次运行：

```bash
python scripts/提取纳入0_20秒波形数据.py
python scripts/可视化纳入0_20秒波形.py
python scripts/批量处理纳入0_20秒_预处理特征.py
python scripts/生成纳入0_20秒_处理结果Excel和Notebook.py
python scripts/生成0_20秒波形聚类分析.py
```

## Notebook

推荐按这个顺序阅读和运行：

1. `notebooks/纳入0-20秒_完整逐步处理与可视化.ipynb`
2. `notebooks/纳入0-20秒_预处理特征可视化.ipynb`
3. `notebooks/纳入0-20秒_聚类分析完整流程.ipynb`

第一份 notebook 从读取数据、预处理、波峰/波谷判断、特征提取到出图完整展开。第三份 notebook 用于复现聚类，包括 K 值评价、PCA、t-SNE 和平均波形图。

## 主要参数

预处理和峰谷判断参数在 `scripts/批量处理纳入0_20秒_预处理特征.py` 中：

```python
BASELINE_RATIO = 0.25
SAVGOL_WINDOW = 51
SAVGOL_POLYORDER = 3
PEAK_PROMINENCE = 0.002
PEAK_HEIGHT = 0.003
PEAK_DISTANCE_SECONDS = 0.30
VALLEY_PROMINENCE = 0.002
VALLEY_DEPTH = 0.003
```

其中 `BASELINE_RATIO = 0.25` 对应 0-20 秒数据的前 5 秒，正好是刺激前基线。Savitzky-Golay 平滑用于降低噪声，同时尽量保留峰的位置和形状。

## 聚类方案

本项目保留两套聚类思路：

1. 解释性特征聚类：使用峰数量、ON 反应幅度、ON 潜伏期、刺激期变化、撤光后变化、AUC、波形标准差和恢复斜率等特征。当前推荐 `K=2`。
2. 完整波形形状聚类：使用每个细胞 0-20 秒整条预处理波形，先 PCA 保留 95% 方差，再聚类。当前推荐 `K=3`。

最终建议优先参考完整波形形状聚类 `K=3`，因为它更符合“按波形来聚类”的目标。

## 主要输出

预处理结果：

- `results/preprocessing/纳入0-20秒_批量预处理特征汇总.xlsx`
- `results/preprocessing/用于聚类的核心特征表.csv`
- `results/preprocessing/单文件独立处理结果/`
- `results/preprocessing/单文件预处理波形图/`

聚类结果：

- `results/clustering/聚类表格/聚类结果_推荐方案.xlsx`
- `results/clustering/聚类表格/K值评价指标.csv`
- `results/clustering/聚类图/PCA_波形聚类_K3.png`
- `results/clustering/聚类图/TSNE_波形聚类_K3.png`
- `results/clustering/聚类图/波形聚类K3_平均波形.png`


