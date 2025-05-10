'''TSI-T的模型结构'''
import torch
import torch.nn as nn
from transformers import BertModel

class BertSelfAttentionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(BertSelfAttentionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结 BERT 参数

        self.attention = nn.Linear(768, 1)  # 简单的注意力机制
        self.fc = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state  # [batch_size, sequence_length, hidden_size]

        # 使用预计算的 mask 进行加权
        mask = mask.unsqueeze(-1).expand_as(x).to(x.device)
        x = x * mask

        # 自注意力层
        attn_scores = self.attention(x).squeeze(-1)  # [batch_size, sequence_length]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, sequence_length]
        attended_features = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size]

        # 分类层
        output = self.fc(self.dropout(attended_features))
        return self.sigmoid(output)
