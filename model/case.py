import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
from new_sense_model import OverallModel,OriginalModel

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

def text_to_tensor(text, tokenizer, keyword_list, p, max_len=320):
    keyword_set = set(keyword_list)
    
    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
    
    # Remove the batch dimension (squeeze) to get (max_len,)
    input_ids = input_ids.squeeze()  # Shape will be (max_len,)
    attention_mask = attention_mask.squeeze()  # Shape will be (max_len,)
    
    # Create mask tensor, initialized to 1s (shape will be (max_len,))
    mask = torch.ones_like(input_ids, dtype=torch.float32)
    
    # Convert input_ids tensor to tokens (list)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())  # Convert to list
    
    # Generate mask based on the keyword list
    for j, token in enumerate(tokens):
        # Replace "##" for subword tokens and check if it matches a keyword
        if token.replace("##", "") in keyword_set:
            mask[j] = 1 + p  # Apply weighting to keyword positions
    
    # Add batch dimension by unsqueezing (shape becomes (1, max_len))
    input_ids = input_ids.unsqueeze(0)  # Shape becomes (1, max_len)
    attention_mask = attention_mask.unsqueeze(0)  # Shape becomes (1, max_len)
    mask = mask.unsqueeze(0)  # Shape becomes (1, max_len)

    return input_ids, attention_mask, mask


'''BKSAT的case study'''

# Main setup and data loading
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
keyword_list={'wanted', 'spot', 'bakery', 'think', 'asked', 'looking', 'stop', 'vegas', 'travel recommendation', 'chip', 'scoop', 'places', 'feel', 'did', 'sugar', 'serving speed', 'came', 'worth trying', 'cheese', 'staff', 'parking', 'hard', 'im', 'actually', 'treats', 'peanut', 'amazing', 'wait', 'perfect', 'fun', 'special', 'value for money', 'outside', 'fresh', 'maybe', 'theyre', 'ambiance', 'small', 'tasty', 'shop', 'visit', 'right', 'absolutely', 'creamy', 'tried', 'scenery', 'lot', 'oh', 'flavors', 'happy', 'butter', 'waffle', 'bit', 'prices', 'town', 'people', 'bouchon', 'pie', 'open', 'stars', 'try', 'away', 'going', 'ice', 'coming', 'delicious', 'sprinkles', 'thought', 'customer', 'order', 'counter', 'sure', 'bread', 'went', 'cookie', 'long', 'cone', 'want', 'food', 'loved', 'taste', 'better', 'portion', 'menu', 'disappointing', 'really', 'pretty', 'location', 'good', 'theres', 'worth', 'ordered', 'line', 'sundae', 'caramel', 'selection', 'treat', 'enjoy', 'crack', 'yummy', 'reservation', 'milk', 'definitely', 'free', 'let', 'cookies', 'cute', 'time', 'sweet', 'probably', 'night', 'birthday', 'store', 'soft', 'friends', 'new', 'friendly', 'signature dish', 'macaroons', 'trying', 'eat', 'cakes', 'say', 'got', 'tea', 'kind', 'french', 'took', 'cuisine', 'thing', 'donut', 'inside', 'awesome', 'strawberry', 'beverages', 'dont', 'cereal', 'cool', 'day', 'cup', 'vanilla', 'ive', 'hot', 'decided', 'know', 'pastries', 'coffee', 'flavor', 'ill', 'thats', 'love', 'negative review', 'just', 'minutes', 'things', 'vegan', 'fan', 'said', 'home', 'dessert', 'place', 'didnt', 'disappointed', 'youre', 'must-visit restaurant', 'friendly staff', 'make', 'cupcake', 'like', 'great', 'macarons', 'seating', 'salted', 'area', 'id', 'best', 'items', 'overall', 'chocolate', 'little', 'times', 'waiting time', 'baked', 'brownie', 'sandwich', 'tableware', 'cream', 'wasnt', 'quality', 'cake', 'experience', 'recommendation', 'need', 'friend', 'nice', 'donuts', 'convenient location', 'service', 'bad', 'bar', 'desserts', 'super', 'way', 'big', 'hygiene', 'unique', 'come', 'overrated', 'different', 'tasted', 'cupcakes', 'recommend', 'red', 'price', 'velvet', 'croissant', 'huge', 'options', 'serve', 'memorable experience', 'favorite', 'getting'}

# 初始化模型并在加载前将模型移到设备上
model = OverallModel(pretrained_dnn_model).to(device)
# 加载模型状态字典
model.load_state_dict(torch.load('./models/newsense_final_model_p0.3_lr0.0001_wd0.0001_final_epoch.pth', map_location=device), strict=False)
# Input texts
texts = ["I can't rate it below than that.", 
         "which airport provide hot snacks only after 0600 Hrs .. I always believed and told my friend you can not go wrong with ITC .. now I am forced to change that believe !!", 
         "Pathetic waiting management. Also they are not as perfect as sarwana bhawan.",
        "Hopeless service!!!  Hopeless service!  It was amazing till it was Biryani Paradise. ",
        "You can have better food in other place will never again go to this place again."]

# Predict class for each sentence
for text in texts:
    # Convert text to tensor
    input_ids, attention_mask, mask = text_to_tensor(text,tokenizer, keyword_list,0.3)
    
    input_ids, attention_mask, mask = input_ids.to(device), attention_mask.to(device), mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, mask)

    # Prediction
    predicted_prob = outputs.item()
    predicted_class = 1 if predicted_prob >= 0.5 else 0

    # Output result
    print(f"Text: '{text}'")
    print(f"Predicted Class: {predicted_class} (0: Negative, 1: Positive)")
    print(f"Probability of Positive: {predicted_prob:.4f}")
    print("-" * 50)
