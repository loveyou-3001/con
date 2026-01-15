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
import copy

# === 引入你的 HOP 模型定义 ===
from hop_model import HOPBertClassifier, get_bert_path
from peft import PeftModel

# === 🌟 论文复现配置 (Paper Reproduction Config) 🌟 ===
DATA_ROOT = "data/dsc_small"        
CKPT_ROOT = "checkpoints_hop_repro" # 区分于之前的文件夹
LOG_FILE = "results_hop_repro.txt"    
NUM_TASKS = 10                      
NUM_CLASSES = 2                     
HOP_ORDER = 1                       # ✅ 论文最佳设置: Mean + Var + Skew
EPOCHS = 5                          
BATCH_SIZE = 8                      
LR = 2e-4                           

warnings.filterwarnings("ignore", category=FutureWarning)

# === 数据集加载器 (保持不变) ===
class JSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(item['label']), dtype=torch.long)
        }

def train_one_task(model, loader, device, task_id):
    # 论文细节：只训练 Adapter (LoRA) 和 MLP Head，冻结 BERT 主干
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"  [Task {task_id}] Epoch {epoch+1}/{EPOCHS}", leave=False)
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
    print(f"🚀 Device: {device} | HOP Order: {HOP_ORDER}")
    os.makedirs(CKPT_ROOT, exist_ok=True)
    
    # 1. 初始化 Tokenizer
    model_path = get_bert_path() 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. ✅ 核心修改：在循环外初始化模型
    # 这样 model 对象在内存中一直存在，Task 1 会自动继承 Task 0 的权重
    print("正在初始化 HOP 模型...")
    model = HOPBertClassifier(
        model_dir=model_path, 
        num_classes=NUM_CLASSES, 
        hop_order=HOP_ORDER, 
        use_lora=True
    ).to(device)

    avg_accuracies = [] # 记录 TIL (Task-Incremental Learning) 的平均准确率

    # === 持续学习主循环 ===
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*15} Training Task: {task_id} {'='*15}")
        
        # --- A. 数据准备 ---
        train_file = os.path.join(DATA_ROOT, f"task_{task_id}", "train.json")
        if not os.path.exists(train_file):
            print(f"❌ 数据缺失: {train_file}")
            continue
        train_loader = DataLoader(JSONLDataset(train_file, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        
        # --- B. 训练 (继承了上一轮的参数) ---
        # 论文点："Initialization of adapters to the last achieved ones"
        # 实现：直接继续训练当前的 model 对象
        train_one_task(model, train_loader, device, task_id)
        
        # --- C. 保存当前任务的专家权重 (Snapshot) ---
        # 我们保存这一刻的状态，用于后续的 TIL 评估
        task_save_dir = os.path.join(CKPT_ROOT, f"task_{task_id}")
        model.bert.save_pretrained(task_save_dir) # 保存 LoRA
        torch.save(model.classifier.state_dict(), os.path.join(task_save_dir, "head.pth")) # 保存 MLP Head
        print(f"💾 Task {task_id} 模型已保存。")

        # --- D. 评估 (TIL Evaluation) ---
        # 论文评估逻辑：评估目前为止所有见过的任务 (0 ~ current task)
        # 关键点：对于 Task k，必须加载当时保存的 Adapter k 和 Head k
        print(f"📊 正在评估 Task 0 ~ {task_id} (TIL Mode)...")
        current_step_accs = []
        
        for eval_id in range(task_id + 1):
            test_file = os.path.join(DATA_ROOT, f"task_{eval_id}", "test.json")
            test_loader = DataLoader(JSONLDataset(test_file, tokenizer), batch_size=32, shuffle=False)
            
            # --- 构造评估专用模型 (避免污染当前训练状态) ---
            # 1. 基础架构
            eval_model = HOPBertClassifier(
                model_dir=model_path, 
                num_classes=NUM_CLASSES, 
                hop_order=HOP_ORDER, 
                use_lora=False # 先不加载 LoRA
            )
            
            # 2. 加载特定任务的 LoRA 权重
            adapter_path = os.path.join(CKPT_ROOT, f"task_{eval_id}")
            eval_model.bert = PeftModel.from_pretrained(
                eval_model.bert, 
                adapter_path, 
                is_trainable=False
            )
            
            # 3. 加载特定任务的 MLP Head 权重
            head_path = os.path.join(CKPT_ROOT, f"task_{eval_id}", "head.pth")
            eval_model.classifier.load_state_dict(torch.load(head_path))
            
            eval_model.to(device)
            
            # 4. 测算
            acc = evaluate(eval_model, test_loader, device)
            current_step_accs.append(acc)
            
            # 5. 释放显存
            del eval_model
            torch.cuda.empty_cache()
            
        # 计算当前阶段的 Average Accuracy
        step_avg = np.mean(current_step_accs)
        avg_accuracies.append(step_avg)
        print(f"👉 Step {task_id} (Seen {task_id+1} tasks) Avg Acc: {step_avg*100:.2f}%")

    # === 最终结果 ===
    final_acc = avg_accuracies[-1] * 100
    print(f"\n🏆 Final DSC (Small) Accuracy: {final_acc:.2f}%")
    print(f"📈 Acc History: {[round(x*100, 2) for x in avg_accuracies]}")
    
    with open(LOG_FILE, "w") as f:
        f.write(f"Results for HOP Reproduction (Order={HOP_ORDER})\n")
        f.write(f"Avg Acc History: {json.dumps(avg_accuracies)}\n")
        f.write(f"Final Acc: {final_acc:.2f}%\n")

if __name__ == "__main__":
    main()