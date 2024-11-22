import torch
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from nltk.tokenize import word_tokenize
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 设置日志记录
logging.basicConfig(filename='new-embedding_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def text_process(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # 移除URL
    text = re.sub(r'@\w+', '', text)  # 移除@提及
    text = re.sub(r'[^A-Za-z\s]', '', text)  # 移除标点符号和数字
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    tokens = word_tokenize(text.lower())  # 分词并转换为小写
    return ' '.join(tokens)  # 将 token 重新组合成字符串

def extract_keywords(texts, top_n=100):
    """
    从文本列表中提取前 `top_n` 个高频关键词。
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # 获取所有文本的关键词并合并为一个集合
    keyword_set = set()
    for row in tfidf_matrix:
        top_indices = row.toarray().argsort()[0][-top_n:]
        top_keywords = feature_names[top_indices]
        keyword_set.update(top_keywords)  
    return keyword_set

def get_keyword(texts):
    # Text preprocessing
    cleaned_texts = [text_process(text) for text in texts]
    # Extract high-frequency keywords
    high_freq_keywords = extract_keywords(cleaned_texts, top_n=100)
    domain_keywords = {
        "cuisine", "taste", "service", "ambiance", "menu", "price", "portion", "beverages", "signature dish",
        "tableware", "waiting time", "reservation", "friendly staff", "hygiene", "serving speed",
        "recommendation", "worth trying", "value for money", "disappointing", "overrated", "memorable experience",
        "negative review", "convenient location", "scenery", "travel recommendation", "must-visit restaurant"
    }
    keyword_list = high_freq_keywords.union(domain_keywords)
    keyword_set = set(keyword_list)
    return keyword_set

class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, p, max_len=320):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.keyword_set = get_keyword(texts)  # 将关键词列表转换为集合，加快匹配速度
        self.p = p
        self.max_len = max_len

        # 提前对所有样本生成掩码矩阵和 BERT 输入
        self.data = self.precompute()

    def precompute(self):
        """
        预处理并缓存掩码矩阵以及 BERT 模型的输入格式。
        """
        data = []
        for text, label in zip(self.texts, self.labels):
            # 对文本进行 BERT tokenization
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze(0)  # [sequence_length]
            attention_mask = encoding['attention_mask'].squeeze(0)

            # 创建掩码矩阵
            mask = torch.ones_like(input_ids, dtype=torch.float32)  # 初始化为 1
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)  # 将 input_ids 转换为 tokens
            
            # 根据关键词列表生成掩码
            for j, token in enumerate(tokens):
                if token.replace("##", "") in self.keyword_set:
                    mask[j] = 1 + self.p  # 对关键词位置进行加权

            # 存储 input_ids、attention_mask、mask 和 label
            data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'mask': mask,
                'label': torch.tensor(label, dtype=torch.float)
            })
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回预处理后的 input_ids、attention_mask、mask 和 label
        item = self.data[idx]
        return item['input_ids'], item['attention_mask'], item['mask'], item['label']
