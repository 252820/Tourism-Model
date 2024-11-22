import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence
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
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # 移除URL
        text = re.sub(r'@\w+', '', text)  # 移除@提及
        text = re.sub(r'[^A-Za-z\s]', '', text)  # 移除标点符号和数字
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        tokens=word_tokenize(text.lower()) #分词
        # 去除停用词
        filtered_tokens=[token for token in tokens if token not in stop_words]
        # 词汇映射
        indices=[vocab[token] if token in vocab else vocab['unk'] for token in filtered_tokens]
        return torch.tensor(indices,dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        label=torch.tensor(self.labels[idx],dtype=torch.float32)
        return text,label

# 处理变长序列，通过填充序列使它们在一个批次内具有相同的长度。  
def collate_fn(batch):  
    texts,labels=zip(*batch)  # 解压批次中的数据
    texts_padded=pad_sequence(texts,batch_first=True,padding_value=0)  # 对文本序列进行填充
    labels=torch.stack(labels)   # 将标签堆叠成一个张量
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
    
param_space = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'weight_decay': [1e-5, 1e-4, 1e-3]
}

def train_one_epoch(model, optimizer, train_loader,valid_loader, device, criterion):
    model.train()
    running_loss = 0.0
    correct_pred = 0
    total_samples = 0
    for texts,labels in tqdm(train_loader):
        texts=texts.to(device)
        labels=labels.to(device)
        outputs=model(texts)
        loss = criterion(outputs.squeeze(), labels.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = (outputs > 0.5).float().squeeze().cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        correct_pred += np.sum(preds == labels_np)
        total_samples += len(labels_np)
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_acc = float(correct_pred) / total_samples

    # 评估模型
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for texts,labels in tqdm(valid_loader):
            texts=texts.to(device)
            labels=labels.to(device)
            outputs=model(texts)
            val_loss = criterion(outputs.squeeze(), labels.squeeze())
            val_running_loss += val_loss.item()
    
    val_loss = val_running_loss / len(valid_loader)
    return epoch_loss, train_acc,val_loss

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts,labels in tqdm(test_loader):
            texts=texts.to(device)
            labels=labels.to(device)
            outputs=model(texts)
            preds = outputs.round().squeeze().cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().detach().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1


def grid_search(model,param_space, train_loader, valid_loader,test_loader, device):
    results = []
    # 获取所有可能的超参数组合
    learning_rates = param_space['learning_rate']
    weight_decays = param_space['weight_decay']

    # 遍历所有超参数组合
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            patience=5
            best_val_loss=float('inf')
            counter=0
            # 初始化模型和优化器
            model.to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # 训练模型
            num_epochs =50
            for epoch in range(num_epochs):
                epoch_loss, train_acc ,val_loss= train_one_epoch(model, optimizer, train_loader, valid_loader,device, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0    
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping after {epoch} epochs without improvement.')
                        break  
                if (epoch+1)%10==0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{epoch_loss:.4f}, Train Acc: {train_acc:.4f}')

            # 评估模型
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
            print(f' Learning Rate: {learning_rate}, Weight Decay: {weight_decay}')
            print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

            # 记录结果
            results.append({
                'accuracy': accuracy,
                'f1': f1,
                'recall':recall,
                'precision':precision,
                'params': {
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay
                }
            })
    # print(results)
    # 根据acc和f1筛选最合适的model
    best_result = sorted(results, key=lambda x: (x['accuracy'], x['f1']), reverse=True)[0]
    return best_result['accuracy'], best_result['f1'], best_result['precision'], best_result['recall'],best_result['params'], best_result

# 分离训练集和测试集
df=pd.read_csv('dnn_dataset.csv')
texts=df['text'].tolist()
labels=df['label'].tolist()
# 分离训练集和测试集
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
valid_texts, test_texts, valid_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)
train_dataset = DataProcessing(train_texts, train_labels, vocab)
valid_dataset = DataProcessing(valid_texts, valid_labels, vocab)
test_dataset = DataProcessing(test_texts, test_labels, vocab)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 使用 GloVe 词向量初始化嵌入层
model=SarcasmDetector(len(glove.stoi),200)
model.embedding.weight.data.copy_(glove.vectors)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_accuracy,best_f1,best_precision,best_recall,best_params,result=grid_search(model,param_space, train_loader, valid_loader,test_loader, device)

print(f'Best Accuracy: {best_accuracy:.4f}')
print(f'Best F1 Score: {best_f1:.4f}')
print(f'Best Precision: {best_precision:.4f}')
print(f'Best Recall: {best_recall:.4f}')
print(f'Best Parameters: {best_params}')
