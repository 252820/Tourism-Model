import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd


# 加载预训练的 GLOVE 词嵌入
glove=GloVe(name='6B',dim=200)
vocab = glove.stoi
# 下载并获取停用词
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 数据预处理
class DataProcessing(Dataset):
    def __init__(self,texts,labels,vocab):
        self.texts=[self._text_to_tensor(text,vocab) for text in texts]
        self.labels=labels

    def _text_to_tensor(self,text,vocab):
        text=re.sub(r'http\S+|www\S+|https\S+', '', text)  # 移除URL
        text = re.sub(r'@\w+', '', text)  # 移除@提及
        # text=re.sub(r'[^\w\s]','',text)  # 去除标点符号
        text = re.sub(r'[^A-Za-z\s]', '', text)  # 移除标点符号和数字
        text=re.sub(r'\s+',' ',text).strip()  # 去除多余空格
        tokens=word_tokenize(text.lower())
        # 去除停用词
        filtered_tokens=[token for token in tokens if token not in stop_words]
        # 词汇映射
        indices=[vocab[token] if token in vocab else vocab['unk'] for token in filtered_tokens]
        return torch.tensor(indices,dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        # label=torch.tensor(self.labels[idx],dtype=torch.float32).unsqueeze(1)
        # 将一维张量转换为二维张量
        label=torch.tensor(self.labels[idx],dtype=torch.float32)
        
        # print(f"Label shape: {label.shape}")
        return text,label
    
def collate_fn(batch):
    texts,labels=zip(*batch)
    texts_padded=pad_sequence(texts,batch_first=True,padding_value=0)
    labels=torch.stack(labels)
    # print(f"Batch labels shape: {labels.shape}")
    # labels=torch.tensor(labels,dtype=torch.float32).unsqueeze(1)
    return texts_padded,labels


class SarcasmDetector(nn.Module):
    def __init__(self,vocab_size,embed_dim):
        super(SarcasmDetector,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)

        self.fc_layers=nn.Sequential(
            nn.Linear(embed_dim*2,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        embeded=self.embedding(x)
        mean_pooled=torch.mean(embeded,dim=1)
        max_pooled,_=torch.max(embeded,dim=1)
        # 将两种池化的结果拼接起来
        pooled=torch.cat((mean_pooled,max_pooled),dim=-1)
        x=self.fc_layers(pooled)
        return x


# 使用 GloVe 词向量初始化嵌入层
model=SarcasmDetector(len(glove.stoi),200)
model.embedding.weight.data.copy_(glove.vectors)


df=pd.read_csv('dnn_dataset.csv')
texts=df['text'].tolist()
labels=df['label'].tolist()
# 划分数据集为训练集、验证集和测试集 6:2:2
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
valid_texts, test_texts, valid_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)
train_dataset = DataProcessing(train_texts, train_labels, vocab)
valid_dataset = DataProcessing(valid_texts, valid_labels, vocab)
test_dataset = DataProcessing(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# 定义损失函数和优化器
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001, weight_decay=1e-5)

# 训练模型
num_epochs=50
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

patience=5
best_val_loss=float('inf')
counter=0

train_losses=[]
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    correct_pred=0
    total_samples = 0
    for texts,labels in tqdm(train_loader):
        texts=texts.to(device)
        labels=labels.to(device)
        outputs=model(texts)
        
        loss=criterion(outputs.squeeze(),labels.squeeze())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 更新预测正确的样本数
        preds = outputs.round().squeeze().cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        correct_pred += np.sum(preds == labels_np)
        total_samples += len(labels_np)
        running_loss+=loss.item()

    epoch_loss=running_loss/len(train_loader)
    train_losses.append(epoch_loss)
    
    # 评估模型
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for texts, labels in tqdm(valid_loader): 
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            val_loss = criterion(outputs.squeeze(), labels.squeeze())
            val_running_loss += val_loss.item()
    
    val_loss = val_running_loss / len(valid_loader)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0    
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping after {epoch} epochs without improvement.')
            break  
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.4f}, Train Acc: {float(correct_pred) / total_samples:.4f}')
    
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs - DNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 测试模型
model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():
    for texts,labels in tqdm(test_loader):
        texts=texts.to(device)
        labels=labels.to(device)
        outputs=model(texts)
        preds=outputs.round().squeeze().cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().detach().numpy())

all_preds=np.array(all_preds).flatten()
all_labels=np.array(all_labels).flatten()

accuracy=accuracy_score(all_labels,all_preds)
precision=precision_score(all_labels,all_preds)
recall=recall_score(all_labels,all_preds)
f1=f1_score(all_labels,all_preds)

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1_score: {f1:.4f}')


# 保存DNN模型的嵌入层和权重
# torch.save(model.embedding.state_dict(),'DNN_embedding_weights.pth')
torch.save(model.state_dict(),'DNN_model_weights.pth')