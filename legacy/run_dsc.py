import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import time
import warnings
import json

# === 引入项目依赖 ===
from hop_model import HOPBertClassifier, get_bert_path
# from news_dataset import NewsDataset  <-- 移除这个引入，我们使用内置的修复版
from peft import PeftModel

# === 🌟 核心实验配置 (DSC-Small) 🌟 ===
DATA_ROOT = "data/dsc_small"        
CKPT_ROOT = "checkpoints_dsc_hop"   
LOG_FILE = "results_dsc_hop.txt"    
NUM_TASKS = 10                      
NUM_CLASSES = 2                     
HOP_ORDER = 1                       # 🔥 HOP-3 (Mean+Var+Skew)
EPOCHS = 5                          
BATCH_SIZE = 8                      
LR = 2e-4                           

warnings.filterwarnings("ignore", category=FutureWarning)

# === 🛠️ 修复版 Dataset 类 (支持 JSONL 格式) ===
class JSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        # 逐行读取 JSONL
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        
        # print(f"    Loaded {len(self.data)} samples from {os.path.basename(path)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = int(item['label'])
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# === 以下训练逻辑保持不变 ===

def train_one_task(model, loader, device):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(input_ids, mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            gts.extend(batch['labels'].cpu().numpy())
    return accuracy_score(gts, preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    print(f"🔥 启动 DSC (Small) 实验 | HOP Order: {HOP_ORDER}")
    
    os.makedirs(CKPT_ROOT, exist_ok=True)
    
    # 自动下载/加载 BERT
    print("正在加载 BERT Tokenizer...")
    model_path = get_bert_path() 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    avg_accuracies = []

    # === 主循环 ===
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*15} Task: {task_id} {'='*15}")
        
        # 1. 加载数据 (使用修复后的 JSONLDataset)
        train_file = os.path.join(DATA_ROOT, f"task_{task_id}", "train.json")
        if not os.path.exists(train_file):
            print(f"❌ 找不到文件: {train_file}")
            return

        train_loader = DataLoader(JSONLDataset(train_file, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. 初始化模型
        model = HOPBertClassifier(
            model_dir=model_path, 
            num_classes=NUM_CLASSES, 
            hop_order=HOP_ORDER, 
            use_lora=True
        ).to(device)
        
        # 3. 训练
        start_time = time.time()
        train_one_task(model, train_loader, device)
        print(f"✅ Task {task_id} 训练完成 ({time.time()-start_time:.1f}s)")
        
        # 4. 保存
        task_save_dir = os.path.join(CKPT_ROOT, f"task_{task_id}")
        model.bert.save_pretrained(task_save_dir)
        torch.save(model.classifier.state_dict(), os.path.join(task_save_dir, "head.pth"))
        
        # 5. 评估
        print(f"📊 正在评估 Task 0 ~ {task_id}...")
        current_step_accs = []
        
        for eval_id in range(task_id + 1):
            test_file = os.path.join(DATA_ROOT, f"task_{eval_id}", "test.json")
            # 使用 JSONLDataset 加载测试数据
            test_loader = DataLoader(JSONLDataset(test_file, tokenizer), batch_size=32, shuffle=False)
            
            # Re-load model
            eval_model = HOPBertClassifier(
                model_dir=model_path, 
                num_classes=NUM_CLASSES, 
                hop_order=HOP_ORDER, 
                use_lora=False 
            )
            eval_model.bert = PeftModel.from_pretrained(
                eval_model.bert, 
                os.path.join(CKPT_ROOT, f"task_{eval_id}"), 
                is_trainable=False
            )
            eval_model.classifier.load_state_dict(torch.load(os.path.join(CKPT_ROOT, f"task_{eval_id}", "head.pth")))
            eval_model.to(device)
            
            acc = evaluate(eval_model, test_loader, device)
            current_step_accs.append(acc)
            
        step_avg = float(np.mean(current_step_accs))
        avg_accuracies.append(step_avg)
        print(f"👉 Step {task_id} Avg Acc: {step_avg*100:.2f}%")

    final_acc = avg_accuracies[-1] * 100
    print(f"\n🏆 Final DSC (Small) Accuracy: {final_acc:.2f}%")
    
    with open(LOG_FILE, "w") as f:
        f.write(f"Avg Acc History: {json.dumps(avg_accuracies)}\n")
        f.write(f"Final Acc: {final_acc:.2f}%\n")
    print(f"📝 结果已保存至 {LOG_FILE}")

if __name__ == "__main__":
    main()