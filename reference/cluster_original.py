# ===================== 1. 导入依赖库 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 核心配置（仅需修改这里） =====================
# 【必填】你的汇总表Excel文件路径
file_path = "E:/YKYJS/视网膜钙离子成像/波形Excel数据(1)/波形Excel数据/result/批量处理汇总总表.xlsx"
# 【可选】聚类结果输出路径
output_path = "E:/YKYJS/视网膜钙离子成像/波形Excel数据(1)/波形Excel数据/result/KMeans聚类结果.xlsx"
# 【可选】可视化图片输出文件夹
plot_output_folder = "E:/YKYJS/视网膜钙离子成像/波形Excel数据(1)/波形Excel数据/result/聚类可视化图"
# 【可选】最优聚类数（None=自动计算，手动填4则直接用4类）
n_clusters = None
# =============================================================================

# 自动创建输出文件夹
import os
os.makedirs(plot_output_folder, exist_ok=True)

# ===================== 3. 数据读取与预处理 =====================
print("📊 开始读取并预处理数据...")
# 读取汇总表
df = pd.read_excel(file_path)
# 查看数据基本信息
print(f"原始数据总行数：{len(df)}，总列数：{len(df.columns)}")
print(f"列名列表：{df.columns.tolist()}")

# ---------------------- 3.1 筛选聚类核心特征（钙成像领域通用有效特征） ----------------------
# 这些特征是和钙信号活性、动力学直接相关的核心指标，无冗余、无共线性
cluster_features = [
    '检测到的钙峰总数',
    '钙峰平均幅度(ΔF/F0)',
    '钙峰平均半高宽(秒)',
    '钙峰平均上升时间(秒)',
    '钙峰平均下降时间(秒)',
    'F/F0全时段标准差',
    '发放频率(个/秒)',
    '信号信噪比'
]

# 检查特征列是否存在，缺失则自动跳过
available_features = [f for f in cluster_features if f in df.columns]
print(f"可用的聚类特征：{available_features}")
if len(available_features) < 2:
    print("❌ 可用特征不足2个，无法进行聚类，请检查列名是否匹配")
    exit()

# ---------------------- 3.2 处理缺失值与异常值 ----------------------
# 提取特征数据
X = df[available_features].copy()
# 处理缺失值：用0填充（无钙峰的样本特征为0）
X = X.fillna(0)
# 处理无穷值
X = X.replace([np.inf, -np.inf], 0)
# 筛选有效样本：排除全0的无效样本
valid_mask = X.sum(axis=1) > 0
X_valid = X[valid_mask]
df_valid = df[valid_mask].reset_index(drop=True)
print(f"有效样本数：{len(X_valid)}（已剔除{len(df)-len(X_valid)}个无效/全0样本）")

# ---------------------- 3.3 数据标准化（K-Means必须步骤，消除量纲影响） ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)
print("✅ 数据标准化完成，已消除不同特征的量纲影响")

# ===================== 4. 自动确定最优聚类数（肘部法则+轮廓系数） =====================
print("\n🔍 开始自动确定最优聚类数...")
# 聚类数测试范围：2-10类
k_range = range(2, 11)
inertia_list = []  # 组内平方和（肘部法则）
silhouette_list = []  # 轮廓系数（越接近1，聚类效果越好）

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    inertia_list.append(kmeans.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, labels))
    print(f"聚类数k={k}，轮廓系数={silhouette_list[-1]:.4f}")

# 自动确定最优聚类数：轮廓系数最高的k
if n_clusters is None:
    best_k_idx = np.argmax(silhouette_list)
    n_clusters = k_range[best_k_idx]
    print(f"\n✅ 自动确定最优聚类数：k={n_clusters}（轮廓系数最高，聚类效果最优）")
else:
    print(f"\n✅ 使用手动指定的聚类数：k={n_clusters}")

# ---------------------- 4.1 绘制最优聚类数分析图 ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# 肘部法则图
ax1.plot(k_range, inertia_list, 'o-', color='#1f77b4', linewidth=2, markersize=8)
ax1.axvline(x=n_clusters, color='red', linestyle='--', label=f'最优聚类数k={n_clusters}')
ax1.set_xlabel('聚类数 k', fontsize=12)
ax1.set_ylabel('组内平方和', fontsize=12)
ax1.set_title('肘部法则确定最优聚类数', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
# 轮廓系数图
ax2.plot(k_range, silhouette_list, 'o-', color='#ff7f0e', linewidth=2, markersize=8)
ax2.axvline(x=n_clusters, color='red', linestyle='--', label=f'最优聚类数k={n_clusters}')
ax2.axhline(y=0, color='gray', linestyle='-')
ax2.set_xlabel('聚类数 k', fontsize=12)
ax2.set_ylabel('轮廓系数', fontsize=12)
ax2.set_title('轮廓系数验证聚类效果', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_folder, '最优聚类数分析图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 最优聚类数分析图已生成并保存")

# ===================== 5. 执行K-Means聚类 =====================
print(f"\n🧩 开始执行K-Means聚类，聚类数k={n_clusters}...")
# 初始化K-Means模型
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
# 执行聚类，获取每个样本的聚类标签
cluster_labels = kmeans.fit_predict(X_scaled)
# 获取聚类中心
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
# 把聚类标签添加到原始数据中
df_valid['聚类编号'] = cluster_labels + 1  # 聚类编号从1开始，更符合阅读习惯
df['聚类编号'] = np.nan
df.loc[valid_mask, '聚类编号'] = cluster_labels + 1
print("✅ 聚类执行完成，已为所有有效样本添加聚类编号")

# ===================== 6. 聚类结果业务解释（核心！给聚类打可理解的标签） =====================
print("\n📋 开始分析聚类特征，生成业务标签...")
# 计算每个聚类的核心特征均值
cluster_stats = []
for i in range(n_clusters):
    cluster_mask = cluster_labels == i
    cluster_data = X_valid[cluster_mask]
    # 计算该聚类的特征均值
    cluster_mean = cluster_data.mean().round(4)
    # 统计该聚类的样本数量
    sample_count = len(cluster_data)
    # 保存统计结果
    cluster_stat = {
        '聚类编号': i+1,
        '样本数量': sample_count,
        '样本占比': f"{sample_count/len(X_valid):.2%}"
    }
    for feature in available_features:
        cluster_stat[feature] = cluster_mean[feature]
    cluster_stats.append(cluster_stat)

# 转为DataFrame
df_cluster_stats = pd.DataFrame(cluster_stats)
print("各聚类核心特征统计：")
print(df_cluster_stats.round(4))

# ---------------------- 6.1 自动生成业务标签（适配钙成像数据场景） ----------------------
# 计算全量样本的特征均值，用于判断高低
feature_means = X_valid.mean()
# 定义业务标签规则
def get_business_label(cluster_idx, cluster_mean):
    label_parts = []
    # 1. 钙峰活性标签（基于钙峰总数/发放频率）
    peak_count = cluster_mean['检测到的钙峰总数']
    fire_freq = cluster_mean['发放频率(个/秒)']
    if peak_count >= feature_means['检测到的钙峰总数'] * 1.5:
        label_parts.append('高钙峰活性')
    elif peak_count <= feature_means['检测到的钙峰总数'] * 0.5:
        label_parts.append('低钙峰活性')
    else:
        label_parts.append('中钙峰活性')
    
    # 2. 动力学标签（基于上升/下降时间、半高宽）
    rise_time = cluster_mean['钙峰平均上升时间(秒)']
    fall_time = cluster_mean['钙峰平均下降时间(秒)']
    fwhm = cluster_mean['钙峰平均半高宽(秒)']
    if rise_time <= feature_means['钙峰平均上升时间(秒)'] * 0.5 and fall_time <= feature_means['钙峰平均下降时间(秒)'] * 0.5:
        label_parts.append('快动力学')
    elif rise_time >= feature_means['钙峰平均上升时间(秒)'] * 1.5 or fall_time >= feature_means['钙峰平均下降时间(秒)'] * 1.5:
        label_parts.append('慢动力学')
    else:
        label_parts.append('正常动力学')
    
    # 3. 异常标签（基于信噪比、标准差）
    snr = cluster_mean['信号信噪比']
    std = cluster_mean['F/F0全时段标准差']
    if snr <= feature_means['信号信噪比'] * 0.3 or std >= feature_means['F/F0全时段标准差'] * 2:
        label_parts.append('异常信号')
    elif fwhm >= feature_means['钙峰平均半高宽(秒)'] * 3:
        label_parts.append('宽峰型')
    
    # 拼接最终标签
    return '-'.join(label_parts)

# 为每个聚类添加业务标签
df_cluster_stats['聚类业务标签'] = df_cluster_stats.apply(
    lambda x: get_business_label(x['聚类编号']-1, x), axis=1
)
# 把业务标签添加到样本数据中
label_map = dict(zip(df_cluster_stats['聚类编号'], df_cluster_stats['聚类业务标签']))
df_valid['聚类业务标签'] = df_valid['聚类编号'].map(label_map)
df['聚类业务标签'] = df['聚类编号'].map(label_map)
print("\n✅ 聚类业务标签生成完成：")
for idx, row in df_cluster_stats.iterrows():
    print(f"聚类{row['聚类编号']}：{row['聚类业务标签']}，样本数{row['样本数量']}")

# ===================== 7. 聚类结果可视化 =====================
print("\n🖼️  开始生成聚类可视化图...")
# ---------------------- 7.1 聚类特征热力图 ----------------------
plt.figure(figsize=(14, 8))
# 标准化聚类中心，用于热力图展示
cluster_centers_scaled = (cluster_centers - feature_means.values) / X_valid.std().values
# 绘制热力图
sns.heatmap(
    cluster_centers_scaled,
    annot=True,
    cmap='coolwarm',
    center=0,
    xticklabels=available_features,
    yticklabels=[f"聚类{i+1}: {label_map[i+1]}" for i in range(n_clusters)],
    fmt='.2f',
    linewidths=0.5
)
plt.title('聚类核心特征热力图（标准化后）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_folder, '聚类特征热力图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 聚类特征热力图已生成并保存")

# ---------------------- 7.2 聚类样本分布饼图 ----------------------
plt.figure(figsize=(10, 10))
plt.pie(
    df_cluster_stats['样本数量'],
    labels=[f"聚类{i+1}: {label_map[i+1]}" for i in range(n_clusters)],
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Set2', n_clusters)
)
plt.title('聚类样本数量分布', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_folder, '聚类样本分布饼图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 聚类样本分布饼图已生成并保存")

# ---------------------- 7.3 PCA降维聚类散点图 ----------------------
# PCA降维到2维，用于可视化
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
# 绘制散点图
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    mask = cluster_labels == i
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"聚类{i+1}: {label_map[i+1]}",
        s=60,
        alpha=0.8,
        edgecolors='none'
    )
plt.xlabel(f'PC1（方差解释率：{pca.explained_variance_ratio_[0]:.2%}）', fontsize=12)
plt.ylabel(f'PC2（方差解释率：{pca.explained_variance_ratio_[1]:.2%}）', fontsize=12)
plt.title('PCA降维聚类结果可视化', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_folder, 'PCA降维聚类散点图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ PCA降维聚类散点图已生成并保存")

# ===================== 8. 结果保存到Excel =====================
print("\n💾 开始保存聚类结果到Excel...")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 1. 全量样本数据+聚类标签
    df.to_excel(writer, sheet_name='全量样本聚类结果', index=False)
    # 2. 聚类核心特征统计
    df_cluster_stats.to_excel(writer, sheet_name='聚类特征统计', index=False)
    # 3. 有效样本详细数据
    df_valid.to_excel(writer, sheet_name='有效样本详细结果', index=False)
print(f"✅ 聚类结果已保存到Excel文件：{output_path}")

# ===================== 9. 最终结果汇总 =====================
print("\n" + "="*80)
print("🎉 K-Means聚类分析全流程完成！")
print(f"📊 总样本数：{len(df)}，有效样本数：{len(X_valid)}")
print(f"🧩 最优聚类数：k={n_clusters}")
print(f"📋 聚类业务标签：")
for idx, row in df_cluster_stats.iterrows():
    print(f"  聚类{row['聚类编号']}：{row['聚类业务标签']}，样本数{row['样本数量']}（{row['样本占比']}）")
print(f"💾 结果Excel文件：{output_path}")
print(f"🖼️  可视化图片文件夹：{plot_output_folder}")
print("="*80)