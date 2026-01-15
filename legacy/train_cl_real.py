import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from hop_model import HOPBertClassifier, get_bert_path
import os
import shutil

# --- 1. 准备真实的微型数据集 (模拟 ASC 任务) ---
# 任务 1: 电子产品 (Laptop)
task1_data = [
    ("The battery life is amazing, lasts all day.", 1), # Positive
    ("Screen resolution is terrible and blurry.", 0),   # Negative
    ("The keyboard feels cheap and sticky.", 0),        # Negative
    ("I love the sleek design and weight.", 1),         # Positive
    ("Boot time is very fast with the SSD.", 1),        # Positive
    ("The fan is too loud when running apps.", 0),      # Negative
] * 10 # 复制多次以便训练

# 任务 2: 餐厅评价 (Restaurant)
task2_data = [
    ("The steak was cooked perfectly, delicious!", 1),  # Positive
    ("Service was slow and the waiter was rude.", 0),   # Negative
    ("Best pasta I have ever had in my life.", 1),      # Positive
    ("The pizza was cold and soggy.", 0),               # Negative
    ("Great atmosphere and friendly staff.", 1),        # Positive
    ("Too expensive for such small portions.", 0),      # Negative
] * 10

class RealTextDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.data = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=32, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_one_task(model, dataloader, device, task_name):
    """训练单个任务的通用函数"""
    print(f"\n>>> 正在训练任务: {task_name}")
    
    # 优化器: 只训练 LoRA 和 Classifier
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(3): # 训练 3 轮
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"  Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc*100:.1f}%")

# --- 主流程：持续学习 ---
def main():
    device = torch.device("cuda")
    model_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 定义保存路径
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # ==========================
    # 阶段 1: 学习 Task A (Laptop)
    # ==========================
    print("\n========== STAGE 1: Learning Laptop Domain ==========")
    # 初始化模型 (HOP Order=3)
    model = HOPBertClassifier(model_path, num_classes=2, hop_order=3, use_lora=True).to(device)
    
    train_loader1 = DataLoader(RealTextDataset(task1_data, tokenizer), batch_size=8, shuffle=True)
    train_one_task(model, train_loader1, device, "Laptop Sentiment")
    
    # 保存 Task 1 的 LoRA 权重
    print("💾 保存 Task 1 模型权重...")
    # PEFT 的 save_pretrained 只会保存 LoRA 权重 (几 MB)，而不是整个 BERT (几百 MB)
    model.bert.save_pretrained(f"{save_dir}/task1_lora")
    torch.save(model.classifier.state_dict(), f"{save_dir}/task1_head.pth")

    # ==========================
    # 阶段 2: 学习 Task B (Restaurant)
    # ==========================
    print("\n========== STAGE 2: Learning Restaurant Domain ==========")
    # 模拟“忘记”：重新加载基座模型
    # HOP 论文策略：为新任务使用一组“新”的 Adapter (LoRA)
    # 所以我们这里重新初始化模型，就会得到一组随机初始化的 LoRA
    model_task2 = HOPBertClassifier(model_path, num_classes=2, hop_order=3, use_lora=True).to(device)
    
    train_loader2 = DataLoader(RealTextDataset(task2_data, tokenizer), batch_size=8, shuffle=True)
    train_one_task(model_task2, train_loader2, device, "Restaurant Sentiment")
    
    print("\n✅ 持续学习流程演示完成！")
    print(f"模型已保存至 {save_dir}/ 目录。")
    print("HOP 结合 LoRA，不仅训练快，而且保存的模型文件极小。")

if __name__ == "__main__":
    main()
