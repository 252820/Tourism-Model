'''BKSAT-T的训练过程'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from Dataset import PrecomputedEmbeddingDataset
from new_BSK import BertSelfAttentionModel
import logging
from transformers import BertTokenizer, BertModel

# 设置日志记录
logging.basicConfig(filename='bsknew-training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Logging of BSKATTN.')

# 训练和验证函数
def train_and_evaluate(p_values, lr_values, weight_decay_values, texts, labels, patience=5, batch_size=128, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 记录最佳结果
    best_val_loss = float('inf')
    best_params = {}
    results = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for p in p_values:
        # 数据集拆分
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
        val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

        # 创建数据集和数据加载器
        train_dataset = PrecomputedEmbeddingDataset(train_texts, train_labels, tokenizer, p)
        val_dataset = PrecomputedEmbeddingDataset(val_texts, val_labels, tokenizer, p)
        test_dataset = PrecomputedEmbeddingDataset(test_texts, test_labels, tokenizer, p)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

        for lr in lr_values:
            for weight_decay in weight_decay_values:
                logging.info(f"Starting training with p={p}, lr={lr}, weight_decay={weight_decay}")
                print(f"Training with p={p}, lr={lr}, weight_decay={weight_decay}")

                # 初始化模型、优化器和损失函数
                model = BertSelfAttentionModel().to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                criterion = nn.BCELoss()

                best_epoch_val_loss = float('inf')
                epochs_no_improve = 0

                # 训练循环
                for epoch in range(num_epochs):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss = validate(model, val_loader, criterion, device)

                    # 每 10 个 epoch 记录一次日志
                    if (epoch + 1) % 10 == 0:
                        print(f"p={p}, lr={lr}, weight_decay={weight_decay}, Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                        logging.info(f"p={p}, lr={lr}, weight_decay={weight_decay}, Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                    # Early Stopping 检查
                    # Early Stopping 检查
                    if val_loss < best_epoch_val_loss:
                        best_epoch_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            # 保存当前最佳模型
                            checkpoint_filename = f"./models/newbsk_best_model_p{p}_lr{lr}_wd{weight_decay}.pth"
                            torch.save(model.state_dict(), checkpoint_filename)
                            logging.info(f"Early stopping. Model checkpoint saved at epoch {epoch+1}")
                            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                            break
                 # 如果没有早停，在最后一个 epoch 检查并保存最佳模型
                final_epoch_filename = f"./models/newbsk_final_model_p{p}_lr{lr}_wd{weight_decay}_final_epoch.pth"
                torch.save(model.state_dict(), final_epoch_filename)
                logging.info(f"Final model saved after training completion with filename {final_epoch_filename}")

                # 测试模型
                accuracy, precision, recall, f1, conf_matrix = test(model, test_loader, device)

                # 打印和记录测试结果
                print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                print("Confusion Matrix:")
                print(conf_matrix)
                logging.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                logging.info(f"Confusion Matrix:\n{conf_matrix}")
                
                # 保存结果
                results.append({
                    'accuracy': accuracy,
                    'f1': f1,
                    'recall': recall,
                    'precision': precision,
                    'params': {'learning_rate': lr, 'weight_decay': weight_decay, 'p': p}
                })

    # 选择最佳结果
    best_result = sorted(results, key=lambda x: (x['accuracy'], x['f1']), reverse=True)[0]
    logging.info(f"Best results: {best_result}")
    return best_result


    
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练模型一个 epoch."""
    model.train()
    running_loss = 0.0
    for input_ids, attention_mask, mask, labels in dataloader:
        input_ids, attention_mask, mask, labels = input_ids.to(device), attention_mask.to(device), mask.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask, mask)
        loss = criterion(outputs.squeeze(), labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证模型并返回平均损失."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, mask, labels in dataloader:
            input_ids, attention_mask, mask, labels = input_ids.to(device), attention_mask.to(device), mask.to(device), labels.to(device)
            
            outputs = model(input_ids, attention_mask, mask)
            val_loss += criterion(outputs.squeeze(), labels).item()
    
    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss


def test(model, dataloader, device):
    """在测试集上评估模型，并返回主要评价指标."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, mask, labels in dataloader:
            input_ids, attention_mask, mask, labels = input_ids.to(device), attention_mask.to(device), mask.to(device), labels.to(device)
            
            outputs = model(input_ids, attention_mask, mask)
            preds = (outputs > 0.5).float().squeeze().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, conf_matrix
