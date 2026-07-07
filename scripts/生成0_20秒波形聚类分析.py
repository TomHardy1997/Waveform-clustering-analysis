from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "preprocessing"
INPUT_FEATURE_CSV = RESULT_DIR / "用于聚类的核心特征表.csv"
INPUT_PREPROCESS_CSV = RESULT_DIR / "单文件CSV数据" / "全部文件_预处理后数据.csv"

OUT_DIR = PROJECT_ROOT / "results" / "clustering"
PLOT_DIR = OUT_DIR / "聚类图"
TABLE_DIR = OUT_DIR / "聚类表格"
NOTEBOOK = OUT_DIR / "纳入0-20秒_聚类分析完整流程.ipynb"
README = OUT_DIR / "聚类分析说明.md"

RANDOM_STATE = 42
K_RANGE = range(2, 11)
RECOMMENDED_FEATURE_K = 2
RECOMMENDED_WAVEFORM_K = 3

FEATURE_COLUMNS = [
    "刺激前是否有波峰",
    "stim_5_10s_peak_count",
    "post_10_20s_valley_count",
    "ON反应幅度",
    "ON反应潜伏期",
    "刺激期相对刺激前变化量",
    "撤光后相对刺激前变化量",
    "刺激期AUC",
    "撤光后AUC",
    "波形标准差",
    "恢复斜率",
]

EXCLUDED_FEATURES = [
    ("刺激前是否有波谷", "所有细胞取值都一样，不能帮助分群。"),
    ("post_10_20s_peak_count", "所有细胞取值都一样，不能帮助分群。"),
    ("OFF反应幅度", "所有细胞取值都为 0，不能帮助分群。"),
    ("OFF反应潜伏期", "所有细胞都是缺失值，不能用于聚类。"),
    ("ON/OFF比值", "OFF 幅度为 0 时会被人为放大，不适合直接聚类。"),
    ("最大波峰幅度", "和 ON 反应幅度在当前数据里基本重复。"),
    ("最大波峰出现时间", "和 ON 反应潜伏期只差 5 秒，信息重复。"),
    ("反应类型初判", "当前全部为 ON 型，不能帮助分群。"),
]


def setup_dirs():
    for folder in [OUT_DIR, PLOT_DIR, TABLE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def prepare_feature_matrix(core):
    feature_data = core[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_scaled = pipe.fit_transform(feature_data)
    return feature_data, x_scaled, pipe


def prepare_waveform_matrix(core, preprocess):
    waveform = preprocess.pivot_table(
        index="file_id",
        columns="时间(秒)",
        values="基线校正后ΔF/F0",
    )
    waveform = waveform.loc[core["file_id"]]
    x_scaled = StandardScaler().fit_transform(waveform.fillna(0))
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_scaled)
    return waveform, x_pca, pca


def evaluate_k(x, name):
    rows = []
    labels_by_k = {}
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(x)
        labels_by_k[k] = labels
        counts = np.bincount(labels)
        rows.append(
            {
                "分析类型": name,
                "K": k,
                "silhouette": silhouette_score(x, labels),
                "calinski_harabasz": calinski_harabasz_score(x, labels),
                "davies_bouldin": davies_bouldin_score(x, labels),
                "最小类样本数": int(counts.min()),
                "各类样本数": "; ".join(f"C{i + 1}={n}" for i, n in enumerate(counts)),
            }
        )
    return pd.DataFrame(rows), labels_by_k


def relabel_by_peak_amp(core, labels):
    tmp = core.copy()
    tmp["_label"] = labels
    order = (
        tmp.groupby("_label")["ON反应幅度"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])


def make_embedding_tables(core, x_feature, x_wave):
    feature_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_feature)
    wave_pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_wave)
    feature_tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(x_feature)
    wave_tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(x_wave)

    emb = core[["file_id", "文件名"]].copy()
    emb["feature_pca1"] = feature_pca[:, 0]
    emb["feature_pca2"] = feature_pca[:, 1]
    emb["feature_tsne1"] = feature_tsne[:, 0]
    emb["feature_tsne2"] = feature_tsne[:, 1]
    emb["waveform_pca1"] = wave_pca2[:, 0]
    emb["waveform_pca2"] = wave_pca2[:, 1]
    emb["waveform_tsne1"] = wave_tsne[:, 0]
    emb["waveform_tsne2"] = wave_tsne[:, 1]
    return emb


def plot_k_metrics(metrics, name, out):
    one = metrics[metrics["分析类型"] == name]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(one["K"], one["silhouette"], marker="o")
    axes[0].set_title("Silhouette higher is better")
    axes[1].plot(one["K"], one["calinski_harabasz"], marker="o", color="#4c78a8")
    axes[1].set_title("Calinski-Harabasz higher is better")
    axes[2].plot(one["K"], one["davies_bouldin"], marker="o", color="#f58518")
    axes[2].set_title("Davies-Bouldin lower is better")
    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{name} clustering K evaluation", y=1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(df, x, y, label_col, title, out):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.scatterplot(data=df, x=x, y=y, hue=label_col, palette="tab10", s=58, ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_waveforms(preprocess, labels_table, label_col, title, out):
    data = preprocess.merge(labels_table[["file_id", label_col]], on="file_id", how="inner")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvspan(5, 10, color="#f2cc68", alpha=0.3, label="Stim 5-10s")
    for cluster_id, one in data.groupby(label_col):
        mean = one.groupby("时间(秒)")["基线校正后ΔF/F0"].mean()
        sem = one.groupby("时间(秒)")["基线校正后ΔF/F0"].sem()
        n = one["file_id"].nunique()
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        e = sem.to_numpy(dtype=float)
        ax.plot(x, y, linewidth=2.2, label=f"C{cluster_id} (n={n})")
        ax.fill_between(x, y - e, y + e, alpha=0.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Baseline corrected ΔF/F0")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_feature_heatmap(core, out):
    corr = core[FEATURE_COLUMNS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Selected feature Spearman correlation")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def make_summary_tables(core, labels_table):
    result = core.merge(labels_table, on=["file_id", "文件名"], how="left")
    feature_cluster_summary = (
        result.groupby("feature_cluster_k2")[FEATURE_COLUMNS]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    waveform_cluster_summary = (
        result.groupby("waveform_cluster_k3")[FEATURE_COLUMNS]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    feature_cluster_summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in feature_cluster_summary.columns
    ]
    waveform_cluster_summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in waveform_cluster_summary.columns
    ]
    return result, feature_cluster_summary, waveform_cluster_summary


def make_readme(metrics, pca):
    best_feature = metrics[metrics["分析类型"] == "feature"].sort_values("silhouette", ascending=False).iloc[0]
    wave_ch = metrics[metrics["分析类型"] == "waveform_pca"].sort_values("calinski_harabasz", ascending=False).iloc[0]
    README.write_text(
        f"""# 0-20 秒波形聚类分析说明

## 本轮采用的两种聚类思路

1. 解释性特征聚类：使用峰数量、ON 反应幅度、ON 潜伏期、刺激期/撤光后变化量、AUC、波形标准差、恢复斜率等 11 个特征。
2. 波形形状聚类：直接使用每个细胞 0-20 秒的完整基线校正曲线，先用 PCA 保留 95% 方差，再做 KMeans 聚类。

## 被排除的特征

{chr(10).join([f'- `{name}`：{reason}' for name, reason in EXCLUDED_FEATURES])}

## 类别数建议

- 解释性特征聚类：Silhouette 最优 K = {int(best_feature['K'])}，本轮保留推荐 `K=2`。
- 波形形状聚类：Calinski-Harabasz 最优 K = {int(wave_ch['K'])}，结合可解释性，本轮推荐 `K=3`。

波形 PCA 保留 95% 方差后使用了 `{pca.n_components_}` 个主成分，累计解释方差为 `{pca.explained_variance_ratio_.sum():.3f}`。

## 主要输出

- `聚类表格/聚类结果_推荐方案.xlsx`
- `聚类表格/K值评价指标.csv`
- `聚类图/特征聚类_K评价.png`
- `聚类图/波形聚类_K评价.png`
- `聚类图/PCA_特征聚类_K2.png`
- `聚类图/TSNE_特征聚类_K2.png`
- `聚类图/PCA_波形聚类_K3.png`
- `聚类图/TSNE_波形聚类_K3.png`
- `聚类图/波形聚类K3_平均波形.png`
- `纳入0-20秒_聚类分析完整流程.ipynb`
""",
        encoding="utf-8",
    )


def make_notebook():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})

    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(True)})

    md(
        """
# 纳入 0-20 秒波形聚类分析完整流程

这份 notebook 用来复现聚类分析。它包含两条线：解释性特征聚类，以及更贴近完整波形形状的 PCA 后聚类。
"""
    )
    code(
        """
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path('/Users/tangdi/Desktop/波形Excel数据/整理后_0-20秒原始Excel')
RESULT_DIR = ROOT / '批量预处理特征结果'
OUT_DIR = RESULT_DIR / '聚类分析结果' / 'notebook逐步运行结果'
OUT_DIR.mkdir(parents=True, exist_ok=True)
"""
    )
    md("## 1. 读取输入表")
    code(
        """
core = pd.read_csv(RESULT_DIR / '用于聚类的核心特征表.csv')
preprocess = pd.read_csv(RESULT_DIR / '单文件CSV数据' / '全部文件_预处理后数据.csv')
print(core.shape, preprocess.shape)
display(core.head())
"""
    )
    md("## 2. 选择用于解释性聚类的特征")
    code(
        f"""
FEATURE_COLUMNS = {json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=4)}
EXCLUDED_FEATURES = {json.dumps(EXCLUDED_FEATURES, ensure_ascii=False, indent=4)}

display(pd.DataFrame(EXCLUDED_FEATURES, columns=['排除特征', '原因']))
feature_data = core[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
display(feature_data.describe())
"""
    )
    md("## 3. 标准化解释性特征")
    code(
        """
feature_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
X_feature = feature_pipe.fit_transform(feature_data)
print(X_feature.shape)
"""
    )
    md("## 4. 准备完整波形矩阵，并用 PCA 压缩")
    code(
        """
waveform = preprocess.pivot_table(index='file_id', columns='时间(秒)', values='基线校正后ΔF/F0')
waveform = waveform.loc[core['file_id']]
X_wave_scaled = StandardScaler().fit_transform(waveform.fillna(0))
wave_pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_wave = wave_pca.fit_transform(X_wave_scaled)
print('原始波形矩阵:', waveform.shape)
print('PCA 后矩阵:', X_wave.shape)
print('累计解释方差:', wave_pca.explained_variance_ratio_.sum())
"""
    )
    md("## 5. 评价不同 K 值")
    code(
        """
def evaluate_k(X, name, k_range=range(2, 11)):
    rows = []
    labels_by_k = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(X)
        labels_by_k[k] = labels
        rows.append({
            '分析类型': name,
            'K': k,
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            '各类样本数': '; '.join([f'C{i+1}={n}' for i, n in enumerate(np.bincount(labels))])
        })
    return pd.DataFrame(rows), labels_by_k

feature_metrics, feature_labels_by_k = evaluate_k(X_feature, 'feature')
wave_metrics, wave_labels_by_k = evaluate_k(X_wave, 'waveform_pca')
metrics = pd.concat([feature_metrics, wave_metrics], ignore_index=True)
display(metrics)
"""
    )
    md("## 6. 画 K 值评价图")
    code(
        """
def plot_k_metrics(metrics, name):
    one = metrics[metrics['分析类型'] == name]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(one['K'], one['silhouette'], marker='o')
    axes[0].set_title('Silhouette higher is better')
    axes[1].plot(one['K'], one['calinski_harabasz'], marker='o')
    axes[1].set_title('Calinski-Harabasz higher is better')
    axes[2].plot(one['K'], one['davies_bouldin'], marker='o')
    axes[2].set_title('Davies-Bouldin lower is better')
    for ax in axes:
        ax.set_xlabel('K')
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig

plot_k_metrics(metrics, 'feature')
plt.show()
plot_k_metrics(metrics, 'waveform_pca')
plt.show()
"""
    )
    md("## 7. 采用推荐方案：特征聚类 K=2，波形聚类 K=3")
    code(
        """
def relabel_by_peak_amp(core, labels):
    tmp = core.copy()
    tmp['_label'] = labels
    order = tmp.groupby('_label')['ON反应幅度'].mean().sort_values().index.tolist()
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])

feature_k = 2
waveform_k = 3
feature_cluster = relabel_by_peak_amp(core, feature_labels_by_k[feature_k])
waveform_cluster = relabel_by_peak_amp(core, wave_labels_by_k[waveform_k])

labels = core[['file_id', '文件名']].copy()
labels['feature_cluster_k2'] = feature_cluster
labels['waveform_cluster_k3'] = waveform_cluster
display(labels.head())
display(labels['feature_cluster_k2'].value_counts().sort_index())
display(labels['waveform_cluster_k3'].value_counts().sort_index())
"""
    )
    md("## 8. PCA 和 t-SNE 可视化")
    code(
        """
feature_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_feature)
wave_pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_wave)
feature_tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=RANDOM_STATE).fit_transform(X_feature)
wave_tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=RANDOM_STATE).fit_transform(X_wave)

emb = labels.copy()
emb['feature_pca1'] = feature_pca[:, 0]
emb['feature_pca2'] = feature_pca[:, 1]
emb['feature_tsne1'] = feature_tsne[:, 0]
emb['feature_tsne2'] = feature_tsne[:, 1]
emb['waveform_pca1'] = wave_pca_2d[:, 0]
emb['waveform_pca2'] = wave_pca_2d[:, 1]
emb['waveform_tsne1'] = wave_tsne[:, 0]
emb['waveform_tsne2'] = wave_tsne[:, 1]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
sns.scatterplot(data=emb, x='feature_pca1', y='feature_pca2', hue='feature_cluster_k2', palette='tab10', ax=axes[0,0])
axes[0,0].set_title('Feature PCA, K=2')
sns.scatterplot(data=emb, x='feature_tsne1', y='feature_tsne2', hue='feature_cluster_k2', palette='tab10', ax=axes[0,1])
axes[0,1].set_title('Feature t-SNE, K=2')
sns.scatterplot(data=emb, x='waveform_pca1', y='waveform_pca2', hue='waveform_cluster_k3', palette='tab10', ax=axes[1,0])
axes[1,0].set_title('Waveform PCA, K=3')
sns.scatterplot(data=emb, x='waveform_tsne1', y='waveform_tsne2', hue='waveform_cluster_k3', palette='tab10', ax=axes[1,1])
axes[1,1].set_title('Waveform t-SNE, K=3')
for ax in axes.ravel():
    ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()
"""
    )
    md("## 9. 查看每类平均波形")
    code(
        """
plot_data = preprocess.merge(labels[['file_id', 'waveform_cluster_k3']], on='file_id', how='inner')
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(0, color='gray', linewidth=1)
ax.axvspan(5, 10, color='#f2cc68', alpha=0.3, label='Stim 5-10s')
for cid, one in plot_data.groupby('waveform_cluster_k3'):
    mean = one.groupby('时间(秒)')['基线校正后ΔF/F0'].mean()
    sem = one.groupby('时间(秒)')['基线校正后ΔF/F0'].sem()
    x = mean.index.to_numpy(dtype=float)
    y = mean.to_numpy(dtype=float)
    e = sem.to_numpy(dtype=float)
    ax.plot(x, y, linewidth=2.2, label=f'C{cid} (n={one.file_id.nunique()})')
    ax.fill_between(x, y-e, y+e, alpha=0.15)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Baseline corrected ΔF/F0')
ax.set_title('Mean waveforms by waveform cluster K=3')
ax.legend()
fig.tight_layout()
plt.show()
"""
    )
    md("## 10. 导出 notebook 运行结果")
    code(
        """
final = core.merge(labels, on=['file_id', '文件名'], how='left').merge(emb, on=['file_id', '文件名', 'feature_cluster_k2', 'waveform_cluster_k3'], how='left')
with pd.ExcelWriter(OUT_DIR / '聚类结果_推荐方案_notebook运行.xlsx', engine='openpyxl') as writer:
    final.to_excel(writer, sheet_name='每个细胞聚类标签', index=False)
    metrics.to_excel(writer, sheet_name='K值评价指标', index=False)
    final.groupby('waveform_cluster_k3')[FEATURE_COLUMNS].mean().to_excel(writer, sheet_name='波形K3特征均值')
    final.groupby('feature_cluster_k2')[FEATURE_COLUMNS].mean().to_excel(writer, sheet_name='特征K2特征均值')
print(OUT_DIR)
"""
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    setup_dirs()
    core = pd.read_csv(INPUT_FEATURE_CSV)
    preprocess = pd.read_csv(INPUT_PREPROCESS_CSV)

    feature_data, x_feature, _ = prepare_feature_matrix(core)
    waveform, x_wave, wave_pca = prepare_waveform_matrix(core, preprocess)

    feature_metrics, feature_labels_by_k = evaluate_k(x_feature, "feature")
    wave_metrics, wave_labels_by_k = evaluate_k(x_wave, "waveform_pca")
    metrics = pd.concat([feature_metrics, wave_metrics], ignore_index=True)

    feature_cluster = relabel_by_peak_amp(core, feature_labels_by_k[RECOMMENDED_FEATURE_K])
    waveform_cluster = relabel_by_peak_amp(core, wave_labels_by_k[RECOMMENDED_WAVEFORM_K])

    labels = core[["file_id", "文件名"]].copy()
    labels["feature_cluster_k2"] = feature_cluster
    labels["waveform_cluster_k3"] = waveform_cluster

    embeddings = make_embedding_tables(core, x_feature, x_wave)
    labels_with_emb = labels.merge(embeddings, on=["file_id", "文件名"], how="left")
    final, feature_summary, waveform_summary = make_summary_tables(core, labels_with_emb)

    plot_feature_heatmap(feature_data, PLOT_DIR / "入选特征相关性热图.png")
    plot_k_metrics(metrics, "feature", PLOT_DIR / "特征聚类_K评价.png")
    plot_k_metrics(metrics, "waveform_pca", PLOT_DIR / "波形聚类_K评价.png")
    scatter_plot(labels_with_emb, "feature_pca1", "feature_pca2", "feature_cluster_k2", "Feature PCA clustering K=2", PLOT_DIR / "PCA_特征聚类_K2.png")
    scatter_plot(labels_with_emb, "feature_tsne1", "feature_tsne2", "feature_cluster_k2", "Feature t-SNE clustering K=2", PLOT_DIR / "TSNE_特征聚类_K2.png")
    scatter_plot(labels_with_emb, "waveform_pca1", "waveform_pca2", "waveform_cluster_k3", "Waveform PCA clustering K=3", PLOT_DIR / "PCA_波形聚类_K3.png")
    scatter_plot(labels_with_emb, "waveform_tsne1", "waveform_tsne2", "waveform_cluster_k3", "Waveform t-SNE clustering K=3", PLOT_DIR / "TSNE_波形聚类_K3.png")
    plot_mean_waveforms(preprocess, labels, "waveform_cluster_k3", "Mean waveforms by waveform cluster K=3", PLOT_DIR / "波形聚类K3_平均波形.png")
    plot_mean_waveforms(preprocess, labels, "feature_cluster_k2", "Mean waveforms by feature cluster K=2", PLOT_DIR / "特征聚类K2_平均波形.png")

    metrics.to_csv(TABLE_DIR / "K值评价指标.csv", index=False, encoding="utf-8-sig")
    labels_with_emb.to_csv(TABLE_DIR / "聚类标签和降维坐标.csv", index=False, encoding="utf-8-sig")
    final.to_csv(TABLE_DIR / "聚类结果_推荐方案.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(TABLE_DIR / "聚类结果_推荐方案.xlsx", engine="openpyxl") as writer:
        final.to_excel(writer, sheet_name="每个细胞聚类标签", index=False)
        metrics.to_excel(writer, sheet_name="K值评价指标", index=False)
        feature_summary.to_excel(writer, sheet_name="特征K2特征统计", index=False)
        waveform_summary.to_excel(writer, sheet_name="波形K3特征统计", index=False)
        pd.DataFrame({"入选聚类特征": FEATURE_COLUMNS}).to_excel(writer, sheet_name="入选特征", index=False)
        pd.DataFrame(EXCLUDED_FEATURES, columns=["排除特征", "原因"]).to_excel(writer, sheet_name="排除特征", index=False)

    make_readme(metrics, wave_pca)
    make_notebook()

    print(f"输出文件夹: {OUT_DIR}")
    print(f"推荐结果: {TABLE_DIR / '聚类结果_推荐方案.xlsx'}")
    print(f"Notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
