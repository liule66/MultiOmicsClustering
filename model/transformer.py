# 导入必要库
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig

# 加载数据集函数
def load_csv(file_path, nrows=None):
    return pd.read_csv(file_path, sep='\t', nrows=nrows, engine='python')

# 加载数据集
gene_expression = load_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', nrows=7000)
print("Size of gene_expression dataset: ", gene_expression.shape)
dna_methylation = load_csv('DNA_methylation_450k', nrows=7000)
print("Size of dna_methylation dataset: ", dna_methylation.shape)
mirna_expression = load_csv('pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena', nrows=700)
print("Size of mirna_expression dataset: ", mirna_expression.shape)
# 选取 clinical 数据
clinical_raw_data = pd.read_csv('Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t', index_col=0)

# 获取每个数据集的列名（样本 ID）
gene_expression_samples = set(gene_expression.columns)
dna_methylation_samples = set(dna_methylation.columns)
miRNA_expression_samples = set(mirna_expression.columns)
clinical_samples = set(clinical_raw_data.index)

# 找到所有数据集中共有的样本 ID
common_samples = gene_expression_samples & dna_methylation_samples & miRNA_expression_samples & clinical_samples

common_samples = list(common_samples)

# 使用共有的样本 ID 来过滤每个数据集
gene_expression_data = gene_expression[common_samples]
dna_methylation_data = dna_methylation[common_samples]
miRNA_expression_data = mirna_expression[common_samples]
clinical_data = clinical_raw_data.loc[common_samples]
print("Size of gene_expression_data: ", gene_expression_data.shape)
print("Size of dna_methylation_data: ", dna_methylation_data.shape)
print("Size of miRNA_expression_data: ", miRNA_expression_data.shape)

print("load dataset Successfully")

# 定义清理函数
def clean_data(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, thresh=0.7*data.shape[0])
    data = data.apply(lambda row: row.fillna(row.mean()), axis=1)
    for column in data.columns:
        upper_limit = data[column].quantile(0.99)
        lower_limit = data[column].quantile(0.01)
        data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data

# 清理每种数据
gene_expression_data = clean_data(gene_expression_data)
dna_methylation_data = clean_data(dna_methylation_data)
mirna_expression_data = clean_data(miRNA_expression_data)
clinical_data['OS.time'] = clinical_data['OS.time'].fillna(clinical_data['last_contact_days_to'])
clinical_data['OS.time'].fillna(clinical_data['OS.time'].mean(), inplace=True)
gene_expression_data = gene_expression_data.iloc[:, :].values.T
dna_methylation_data = dna_methylation_data.iloc[:, :].values.T
mirna_expression_data = mirna_expression_data.iloc[:, :].values.T

print("Size of gene_expression_data: ", gene_expression_data.shape)
print("Size of dna_methylation_data: ", dna_methylation_data.shape)
print("Size of miRNA_expression_data: ", mirna_expression_data.shape)

scaler = StandardScaler()
gene_expression_scaled = scaler.fit_transform(gene_expression_data)
dna_methylation_scaled = scaler.fit_transform(dna_methylation_data)
mirna_expression_scaled = scaler.fit_transform(mirna_expression_data)

# 使用PCA进行降维
pca_gene = PCA(n_components=1000)
# Adjusting n_components for dna_methylation PCA
pca_dna = PCA(n_components=min(7000, dna_methylation_data.shape[1]))
pca_mirna = PCA(n_components=100)

print("PCA step completed")

gene_expression_pca = pca_gene.fit_transform(gene_expression_scaled)
dna_methylation_pca = pca_dna.fit_transform(dna_methylation_scaled)
mirna_expression_pca = pca_mirna.fit_transform(mirna_expression_scaled)

# 数据整合
integrated_data = np.concatenate([gene_expression_pca, dna_methylation_pca, mirna_expression_pca], axis=1)
print("Size of integrated_data: ", integrated_data.shape)

# 转换为张量
X_integrated = torch.tensor(integrated_data, dtype=torch.float32)
y_dummy = torch.tensor(np.random.randint(0, 5, integrated_data.shape[0]), dtype=torch.long)

# 创建数据加载器
dataset = TensorDataset(X_integrated, y_dummy)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 定义Transformer模型
class MultiOmicsTransformer(nn.Module):
    def __init__(self):
        super(MultiOmicsTransformer, self).__init__()
        config = BertConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=6,
            intermediate_size=2048,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.transformer = BertModel(config)
        self.fc = nn.Linear(512, 2100)

    def forward(self, x):
        # Transformer expects input of shape [batch_size, sequence_length, hidden_size]
        x = x.unsqueeze(1)  # Adding a dummy sequence length dimension
        outputs = self.transformer(x).last_hidden_state
        outputs = outputs.squeeze(1)  # Remove the dummy sequence length dimension
        outputs = self.fc(outputs)
        return outputs

# 使用改进的模型
model = MultiOmicsTransformer().to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 定义重构损失函数
def reconstruction_loss(reconstructed, original):
    mse_loss = nn.MSELoss()
    return mse_loss(reconstructed, original)

# 训练模型
num_epochs = 50
prev_loss = float('inf')
loss_list = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for X, _ in dataloader:
        X = X.to(device)
        outputs = model(X)
        loss = reconstruction_loss(outputs, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    loss_list.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 在整个数据集上重新计算模型的输出
all_outputs = []
model.eval()  # 切换到评估模式
with torch.no_grad():
    for X, _ in dataloader:
        X = X.to(device)
        output_batch = model(X)
        all_outputs.append(output_batch.cpu().numpy())

# 合并所有批次的输出
all_outputs = np.concatenate(all_outputs, axis=0)
print("Size of all_outputs: ", all_outputs.shape)

# 使用聚类算法进行聚类分析
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(all_outputs)

# 结果评估（轮廓系数）
silhouette_avg = silhouette_score(all_outputs, labels)
print(f'Silhouette Score: {silhouette_avg}')

# 使用PCA降维
pca = PCA(n_components=2)
outputs_2d = pca.fit_transform(all_outputs)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(outputs_2d[:, 0], outputs_2d[:, 1], c=labels)
plt.title('Cluster Visualization')
plt.savefig('cluster_visualization.png')
plt.close()

# 绘制训练的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.close()

def find_optimal_k_and_plot(survival_data, combined_data, k_range):
    optimal_k = None
    best_p_value = 1
    best_labels = None
    total_p_values = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(all_outputs)

        survival_data['Cluster'] = labels
        kmf = KaplanMeierFitter()
        survival_curves = []

        for cluster in np.unique(labels):
            cluster_data = survival_data[survival_data['Cluster'] == cluster]
            T = cluster_data['OS.time']
            E = np.where(cluster_data['vital_status'] == 'Alive', 1, 0)
            kmf.fit(T, event_observed=E)
            survival_curves.append((T, E))

        p_values = []
        for i in range(len(survival_curves) - 1):
            for j in range(i + 1, len(survival_curves)):
                result = logrank_test(survival_curves[i][0], survival_curves[j][0],
                                      event_observed_A=survival_curves[i][1], event_observed_B=survival_curves[j][1])
                p_values.append(result.p_value)

        min_p_value = np.mean(p_values) if p_values else 1
        total_p_values.append(min_p_value)
        if min_p_value < best_p_value:
            best_p_value = min_p_value
            optimal_k = k
            best_labels = labels

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, total_p_values, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('P values')
    plt.title('P values for Different k')
    plt.savefig('p_values.png')
    plt.close()

    if optimal_k:
        print(f"Optimal k: {optimal_k} with p-value: {best_p_value}")
        survival_data['Cluster'] = best_labels
        kmf = KaplanMeierFitter()

        plt.figure(figsize=(10, 6))
        for cluster in np.unique(best_labels):
            cluster_data = survival_data[survival_data['Cluster'] == cluster]
            T = cluster_data['OS.time']
            E = np.where(cluster_data['vital_status'] == 'Alive', 1, 0)
            kmf.fit(T, event_observed=E, label=f'Cluster {cluster}')
            kmf.plot_survival_function()

        plt.title('Survival Analysis by Cluster with Optimal k')
        plt.savefig('survival_analysis.png')
        plt.close()
    else:
        print("No optimal k found.")

find_optimal_k_and_plot(clinical_data, all_outputs, range(2, 10))
