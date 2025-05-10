from new_train import train_and_evaluate
from new_sense_model import  OriginalModel
import torch
import torch.nn as nn
import pandas as pd


p_values = [0.1, 0.3, 0.5,0.8,1]
lr_values = [1e-4, 5e-5]
weight_decay_values = [1e-4, 1e-5]

# Step 1: 生成不同 p 值的嵌入文件
df = pd.read_csv('Yelp-merged.csv')
texts = df['Review Text'].tolist()
labels = df['sentiment'].tolist()

# df = pd.read_csv('Trip-merged.csv')
# texts = df['review_full'].tolist()
# labels = df['sentiment'].tolist()
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

# Step 2: 使用不同的超参数组合训练和验证模型
train_and_evaluate(p_values, lr_values, weight_decay_values,pretrained_dnn_model,texts,labels)





