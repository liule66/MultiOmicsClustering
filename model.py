# 导入必要库
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# 加载数据集函数
def load_gzip_csv(file_path):
    with gzip.open(file_path, 'rt') as f:
        return pd.read_csv(f, sep='\t', engine='python')


# 加载数据集
gene_expression = load_gzip_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz')
dna_methylation = load_gzip_csv(
    'jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz')
mirna_expression = load_gzip_csv('pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz')
clinical_data = pd.read_csv('Survival_SupplementalTable_S1_20171025_xena_sp')

# 数据预处理
# 只使用数值列进行标准化
gene_expression_data = gene_expression.iloc[1:, 1:].values
dna_methylation_data = dna_methylation.iloc[1:, 1:].values
mirna_expression_data = mirna_expression.iloc[1:, 1:].values

scaler = StandardScaler()
gene_expression_scaled = scaler.fit_transform(gene_expression_data)
dna_methylation_scaled = scaler.fit_transform(dna_methylation_data)
mirna_expression_scaled = scaler.fit_transform(mirna_expression_data)

# PCA降维
pca = PCA(n_components=50)
gene_expression_pca = pca.fit_transform(gene_expression_scaled)
dna_methylation_pca = pca.fit_transform(dna_methylation_scaled)
mirna_expression_pca = pca.fit_transform(mirna_expression_scaled)

# 数据整合
integrated_data = np.concatenate([gene_expression_pca, dna_methylation_pca, mirna_expression_pca], axis=1)

# 转换为张量
X_gene = torch.tensor(gene_expression_pca, dtype=torch.float32)
X_dna = torch.tensor(dna_methylation_pca, dtype=torch.float32)
X_mirna = torch.tensor(mirna_expression_pca, dtype=torch.float32)
y_dummy = torch.tensor(np.random.randint(0, 5, integrated_data.shape[1]), dtype=torch.long)

# 创建数据加载器
dataset = TensorDataset(X_gene, X_dna, X_mirna, y_dummy)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


criterion = nn.CrossEntropyLoss()
# 定义模型
# 定义模型
class MultiOmicsNN(nn.Module):
    def __init__(self):
        super(MultiOmicsNN, self).__init__()
        self.fc_gene = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_dna = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mirna = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(16 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x_gene, x_dna, x_mirna):
        x_gene_encoded = self.fc_gene(x_gene)
        x_dna_encoded = self.fc_dna(x_dna)
        x_mirna_encoded = self.fc_mirna(x_mirna)
        x_combined = torch.cat((x_gene_encoded, x_dna_encoded, x_mirna_encoded), dim=1)
        output = self.fc_combined(x_combined)
        return output, x_combined

model = MultiOmicsNN().to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 定义重构损失函数
def reconstruction_loss(reconstructed, original):
    mse_loss = nn.MSELoss()
    return mse_loss(reconstructed, original)

# 训练模型
num_epochs = 100
prev_loss = float('inf')
loss_list = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for gene, dna, mirna, labels in dataloader:
        gene, dna, mirna, labels = gene.to(device), dna.to(device), mirna.to(device), labels.to(device)

        # 前向传播
        outputs, reconstructed = model(gene, dna, mirna)
        loss = reconstruction_loss(reconstructed, torch.cat((gene, dna, mirna), dim=1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    loss_list.append(epoch_loss)

    # 早停法
    if epoch > 0 and epoch_loss > prev_loss:
        print(f'Early stopping at epoch {epoch}')
        break
    prev_loss = epoch_loss

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 使用聚类算法进行聚类分析
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(integrated_data)

# 结果评估（轮廓系数）
silhouette_avg = silhouette_score(integrated_data, labels)
print(f'Silhouette Score: {silhouette_avg}')

# 生存分析
# 这部分代码是进行生存分析的。生存分析是一种统计方法，用于分析预期生存时间或持续时间，常用于医学研究。
#
# 首先，代码将聚类结果（`labels`）添加到临床数据（`clinical_data`）中，作为一个新的列`'cluster'`。
#
# 然后，创建了一个KaplanMeierFitter对象`kmf`，这是一个用于进行Kaplan-Meier生存估计的类。Kaplan-Meier方法是一种非参数统计方法，用于估计从时间序列观察开始到某个事件发生的时间的生存函数。
#
# 接下来，代码遍历每一个独特的聚类（`cluster`），并从临床数据中选取出属于该聚类的数据（`cluster_data`）。然后，使用`kmf.fit()`方法对每个聚类的数据进行生存分析。`fit()`方法的第一个参数是观察的持续时间，第二个参数是观察的事件发生状态（在这个例子中，事件是患者的生存状态）。
#
# 最后，使用`kmf.plot_survival_function()`方法绘制每个聚类的生存函数图。这个图可以用来观察不同聚类的生存情况，比如某个聚类的患者是否比其他聚类的患者有更长的生存时间。
clinical_data['cluster'] = labels

kmf = KaplanMeierFitter()

for cluster in clinical_data['cluster'].unique():
    cluster_data = clinical_data[clinical_data['cluster'] == cluster]
    kmf.fit(cluster_data['Overall_Survival_(months)'], event_observed=cluster_data['Overall_Survival_Status'])
    kmf.plot_survival_function(label=f'Cluster {cluster}')

import matplotlib.pyplot as plt

# 绘制训练的损失曲线
plt.plot(loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
