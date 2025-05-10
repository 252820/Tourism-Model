import torch
import torch.nn as nn
import pandas as pd
from BSK_train import train_and_evaluate

p_values = [0.1, 0.3,0.5,0.8,1]
lr_values = [1e-4, 5e-5]
weight_decay_values = [1e-4, 1e-5]

# Step 1: 生成不同 p 值的嵌入文件
# df = pd.read_csv('Yelp-merged.csv')
# texts = df['Review Text'].tolist()
# labels = df['sentiment'].tolist()

df = pd.read_csv('Trip-merged.csv')
texts = df['review_full'].tolist()
labels = df['sentiment'].tolist()

# Step 2: 使用不同的超参数组合训练和验证模型
train_and_evaluate(p_values, lr_values, weight_decay_values,texts,labels)





