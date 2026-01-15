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

# === 引入依赖 ===
from hop_model import HOPBertClassifier, get_bert_path
from peft import PeftModel
# 🔥 引入 Sleep 模块
try:
    from sleep import sleep_phase
    print("✅ 成功加载 Sleep 模块")
except ImportError:
    print("❌ 未找到 sleep.py，请先创建该文件！")
    exit()

# === 🌟 Sleep 实验配置 🌟 ===
DATA_ROOT = "data/dsc_small"
CKPT_ROOT = "checkpoints_sleep"       # 区分目录
LOG_FILE = "results_sleep.txt"
NUM_TASKS = 10
NUM_CLASSES = 2
HOP_ORDER = 1                         # 🔥 使用你发现的最强设置 (Order=1)
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-4

# === 💤 Sleep 超参数 (这是你要调优的重点) ===
SLEEP_CONFIG = {
    'alpha': 0.02,        # 🔥 极低衰减：只减弱 2% (微创)
    'beta': 0.02,         # 🔥 净保持：核心参数 1 - 0.02 + 0.02 = 1.0 (完全不变)
    'mask_threshold': 20  # 🔥 高门槛：只保护 Top 20% 的精英参数，其余 80% 都会被削弱
}

warnings.filterwarnings("ignore", category=FutureWarning)

# (Dataset 类保持不变，为了节省篇幅省略，直接复制之前的即可，或者如下)
class JSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(int(item['label']), dtype=torch.long)}

def train_one_task(model, loader, device, task_id):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"  [Task {task_id}] Epoch {epoch+1}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            gts.extend(batch['labels'].cpu().numpy())
    return accuracy_score(gts, preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | HOP Order: {HOP_ORDER} | Mode: Sleep Experiment")
    os.makedirs(CKPT_ROOT, exist_ok=True)
    
    model_path = get_bert_path() 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 初始化模型 (参数继承)
    print("正在初始化 HOP 模型...")
    model = HOPBertClassifier(model_path, NUM_CLASSES, HOP_ORDER, use_lora=True).to(device)

    avg_accuracies = []

    for task_id in range(NUM_TASKS):
    #for task_id in range(3):
        print(f"\n{'='*15} Task: {task_id} (Wake Phase) {'='*15}")
        
        # 1. Wake Phase: 正常训练
        train_file = os.path.join(DATA_ROOT, f"task_{task_id}", "train.json")
        if not os.path.exists(train_file): continue
        train_loader = DataLoader(JSONLDataset(train_file, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        
        train_one_task(model, train_loader, device, task_id)
        
        # 2. 💤 Sleep Phase: 突触缩减 (在保存模型之前！) 💤
        # 我们希望保存下来的是经过"睡眠整理"后的更干净的参数
        model = sleep_phase(model, tokenizer, device, SLEEP_CONFIG)
        
        # 3. 保存模型 (Snapshot)
        task_save_dir = os.path.join(CKPT_ROOT, f"task_{task_id}")
        model.bert.save_pretrained(task_save_dir)
        torch.save(model.classifier.state_dict(), os.path.join(task_save_dir, "head.pth"))
        print(f"💾 Task {task_id} 模型已保存 (Sleeped).")

        # 4. 评估 (TIL)
        # 注意：这里我们评估的是 Sleep 后的模型。
        # 如果 Sleep 导致精度大幅下降，说明 alpha 太大或 Mask 不准。
        # 理想情况：精度保持不变，但参数范数变小 (已在 sleep_phase 打印)。
        print(f"📊 正在评估 Task 0 ~ {task_id}...")
        current_step_accs = []
        for eval_id in range(task_id + 1):
            test_file = os.path.join(DATA_ROOT, f"task_{eval_id}", "test.json")
            test_loader = DataLoader(JSONLDataset(test_file, tokenizer), batch_size=32, shuffle=False)
            
            # 构造评估模型
            eval_model = HOPBertClassifier(model_path, NUM_CLASSES, HOP_ORDER, use_lora=False)
            eval_model.bert = PeftModel.from_pretrained(eval_model.bert, os.path.join(CKPT_ROOT, f"task_{eval_id}"), is_trainable=False)
            eval_model.classifier.load_state_dict(torch.load(os.path.join(CKPT_ROOT, f"task_{eval_id}", "head.pth")))
            eval_model.to(device)
            
            acc = evaluate(eval_model, test_loader, device)
            current_step_accs.append(acc)
            del eval_model
            torch.cuda.empty_cache()
            
        step_avg = float(np.mean(current_step_accs))
        avg_accuracies.append(step_avg)
        print(f"👉 Step {task_id} Avg Acc: {step_avg*100:.2f}%")

    final_acc = avg_accuracies[-1] * 100
    print(f"\n🏆 Final Sleep Accuracy: {final_acc:.2f}%")
    print(f"📈 History: {[round(x*100, 2) for x in avg_accuracies]}")
    
    with open(LOG_FILE, "w") as f:
        f.write(f"Results for Sleep Experiment (Order={HOP_ORDER}, Config={SLEEP_CONFIG})\n")
        f.write(f"Avg Acc History: {json.dumps(avg_accuracies)}\n")
        f.write(f"Final Acc: {final_acc:.2f}%\n")

if __name__ == "__main__":
    main()