# MultiOmicsClustering

## 项目简介

本项目旨在通过整合多组学数据（基因表达、DNA 甲基化、miRNA 表达）进行聚类分析，使用神经网络模型来处理降维后的数据，并通过生存分析评估聚类结果的生物学意义。

---

## 项目结构

data/：存放数据集文件。

models/：存放神经网络模型定义文件。

notebooks/：存放 Jupyter notebook 文件。

scripts/：存放数据处理和训练脚本文件。

results/：存放结果文件，包括训练损失曲线、聚类可视化和生存分析图。

---
## 安装依赖

首先，确保你已安装 Python 及其包管理工具 pip。然后，可以通过以下命令安装所需的库：

bash pip install -r requirements.txt

其中 requirements.txt 文件包含如下内容：

numpy

pandas

torch

lifelines

scikit-learn

matplotlib

---

## 数据集

数据集文件需放置在 data/ 目录下，具体包括：


EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena

DNA_methylation_450k

pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena

Survival_SupplementalTable_S1_20171025_xena_sp

## 运行

python model2.py
