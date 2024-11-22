import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
from new_sense_model import OverallModel,OriginalModel
from Dataset import PrecomputedEmbeddingDataset
from new_train import test

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练模型权重
pretrained_weights_path = 'DNN_model_weights.pth'
pretrained_weights = torch.load(pretrained_weights_path)
# 创建原始模型
original_model = OriginalModel()
# 加载预训练权重
original_model.load_state_dict(pretrained_weights)
# 移除最后1层
pretrained_dnn_model = nn.Sequential(*list(original_model.fc_layers.children())[:-1])
print(pretrained_dnn_model)

# Load the dataset
df = pd.read_csv('irony.csv')
texts = df['review'].tolist()
labels = df['sentiment'].tolist()
# Create datasets and dataloaders
irony_dataset = PrecomputedEmbeddingDataset(texts, labels, tokenizer, 0.3)
batch_size = 128
irony_loader = DataLoader(irony_dataset, batch_size=batch_size,  shuffle=True, num_workers=12, pin_memory=True)

# 初始化模型并在加载前将模型移到设备上
model = OverallModel(pretrained_dnn_model).to(device)
# 加载模型状态字典
model.load_state_dict(torch.load('./models/newsense_final_model_p0.3_lr0.0001_wd0.0001_final_epoch.pth', map_location=device), strict=False)

accuracy, precision, recall, f1,confusion_matrix = test(model, irony_loader, device)

# 提取混淆矩阵中的值
TN, FP, FN, TP = confusion_matrix.ravel()

# 计算 NPV
npv = TN / (TN + FN)

print("Confusion Matrix:")
print(confusion_matrix)
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
print(f'Negative Predictive Value (NPV): {npv:.4f}')