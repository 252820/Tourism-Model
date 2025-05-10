'''TSI的模型结构'''
import torch
import torch.nn as nn
from transformers import BertModel

class TourismKeywordWeightAdjustment(nn.Module):
    """
    用于 BERT 嵌入和关键词权重调整的类。
    """
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(TourismKeywordWeightAdjustment, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结 BERT 参数

    def forward(self, input_ids, attention_mask, mask):
        # 获取 BERT 嵌入
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state  # [batch_size, sequence_length, hidden_size]

        # 使用预计算的关键词 mask 进行加权
        mask = mask.unsqueeze(-1).expand_as(x).to(x.device)
        weighted_x = x * mask  # [batch_size, sequence_length, hidden_size]
        
        return weighted_x


class TextWordAttentionCalculation(nn.Module):
    """
    用于关键词加权后嵌入的注意力机制计算。
    """
    def __init__(self, input_dim=768):
        super(TextWordAttentionCalculation, self).__init__()
        self.attention = nn.Linear(input_dim, 1)  # 简单的注意力机制

    def forward(self, x):
        # 自注意力层
        attn_scores = self.attention(x).squeeze(-1)  # [batch_size, sequence_length]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, sequence_length]
        attended_features = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, input_dim]
        
        return attended_features



class SatiricalKnowledgeTransferLearning(nn.Module):
    def __init__(self, pretrained_dnn_model, freeze_dnn=False):
        super(SatiricalKnowledgeTransferLearning, self).__init__()
        self.dnn_model = pretrained_dnn_model
        if freeze_dnn:
            for param in self.dnn_model.parameters():
                param.requires_grad = False
        self.reduce_dim = nn.Sequential(
            nn.Linear(768, 400),  # 确保输入为 768 维
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 确保输入 x 的形状为 [batch_size, sequence_length, 768]
        # print(f"x shape before mean pooling: {x.shape}")  torch.Size([128, 768])
        # mean_pooled_embedding = torch.mean(x, dim=1)  # 应该得到 [batch_size, 768]
        # print(f"mean_pooled_embedding shape after mean pooling: {mean_pooled_embedding.shape}")
    
        # 通过降维层
        reduced_embedding = self.reduce_dim(x)  # 应该输出 [batch_size, 400]
        # print(f"reduced_embedding shape after reduce_dim: {reduced_embedding.shape}")
        
        # 输入到预训练 DNN 模型
        dnn_output = self.dnn_model(reduced_embedding)
        return dnn_output



class SentimentClassification(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassification, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class OverallModel(nn.Module):
    def __init__(self, pretrained_dnn_model, freeze_dnn=False):
        super(OverallModel, self).__init__()
        self.keyword_adjustment=TourismKeywordWeightAdjustment()
        self.attention_calculation = TextWordAttentionCalculation()
        self.knowledge_transfer_learning = SatiricalKnowledgeTransferLearning(pretrained_dnn_model=pretrained_dnn_model)
        dnn_output_dim = 1
        attn_output_dim = 768  # BERT 默认的嵌入维度
        total_concat_dim = dnn_output_dim + attn_output_dim
        self.sentiment_classification = SentimentClassification(total_concat_dim)

    def forward(self, input_ids, attention_mask, mask):
        weighted_x = self.keyword_adjustment(input_ids, attention_mask, mask)
        # Step 2: 自注意力机制
        attention_output = self.attention_calculation(weighted_x)
        # Satirical Knowledge Transfer Learning
        transfer_learning_output = self.knowledge_transfer_learning(attention_output)

        # Concatenate outputs before final sentiment classification
        concat_output = torch.cat([transfer_learning_output, attention_output], dim=-1)

        # Sentiment Classification
        sentiment_output = self.sentiment_classification(concat_output)

        return sentiment_output

class OriginalModel(nn.Module):
    def __init__(self, vocab_size=400000, embed_dim=200):
        super(OriginalModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embeded = self.embedding(x)
        mean_pooled = torch.mean(embeded, dim=1)
        max_pooled, _ = torch.max(embeded, dim=1)
        # 将两种池化的结果拼接起来
        pooled = torch.cat((mean_pooled, max_pooled), dim=-1)
        x = self.fc_layers(pooled)
        return x