import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from hop_model import HOPBertClassifier, get_bert_path
import numpy as np
import time

# --- 1. 模拟数据集 (无需下载真实数据即可验证) ---
class DummyDataset(Dataset):
    def __init__(self, tokenizer, num_samples=200):
        self.tokenizer = tokenizer
        self.data = []
        # 模拟 3 分类任务
        for i in range(num_samples):
            text = f"This is a sample sentence number {i} to test High Order Pooling logic."
            label = np.random.randint(0, 3)
            self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=64, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 2. 主流程 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行设备: {device}")

    # 获取模型和 Tokenizer
    model_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 初始化模型
    print("正在初始化 HOP 模型 (Order=3, LoRA enabled)...")
    model = HOPBertClassifier(
        model_dir=model_path, 
        num_classes=3, 
        hop_order=3, 
        use_lora=True
    ).to(device)

    # 准备数据
    print("📦 生成模拟数据...")
    train_dataset = DummyDataset(tokenizer, num_samples=128)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 优化器 (只训练 LoRA 和 MLP 头)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print("\n🔥 开始训练 (Epochs=3)...")
    model.train()
    
    for epoch in range(3):
        start = time.time()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {time.time()-start:.2f}s")

    print("\n✅ HOP 逻辑验证成功！模型可以正常前向和反向传播。")

if __name__ == "__main__":
    main()
