import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import time
import warnings
import json 

# === 引入依赖 ===
# 确保 hop_model.py 和 news_dataset.py 在同一目录下
from hop_model import HOPBertClassifier, get_bert_path
from news_dataset import NewsDataset
from peft import PeftModel

# === 🚨 Baseline 核心配置 🚨 ===
# 1. 存储路径区分开，避免覆盖 HOP 的结果
CKPT_ROOT = "checkpoints_baseline"  
# 2. 关键修改：设置为 1 阶，即普通的 Mean Pooling
HOP_ORDER = 1                       
# 3. 结果日志区分开
LOG_FILE = "results_baseline.txt"   

# === 其他配置保持一致 ===
DATA_ROOT = "data/20news"
NUM_TASKS = 10
NUM_CLASSES = 2
EPOCHS = 3
BATCH_SIZE = 16
LR = 5e-4

warnings.filterwarnings("ignore", category=FutureWarning)

def train_one_task(model, loader, device):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"  [Baseline] Train Epoch {epoch+1}/{EPOCHS}", leave=False)
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
    print(f"📉 启动 Baseline 实验 (Mean Pooling, Order={HOP_ORDER})")
    
    os.makedirs(CKPT_ROOT, exist_ok=True)
    model_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    avg_accuracies = []

    # === 主循环 ===
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*10} Baseline Task: {task_id} / {NUM_TASKS-1} {'='*10}")
        
        # 1. 数据
        train_file = os.path.join(DATA_ROOT, f"task_{task_id}", "train.json")
        train_loader = DataLoader(NewsDataset(train_file, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. 模型初始化 (HOP_ORDER = 1)
        model = HOPBertClassifier(
            model_dir=model_path, 
            num_classes=NUM_CLASSES, 
            hop_order=HOP_ORDER, # <--- 关键点
            use_lora=True
        ).to(device)
        
        # 3. 训练
        start_time = time.time()
        train_one_task(model, train_loader, device)
        print(f"✅ Baseline Task {task_id} Done. ({time.time()-start_time:.1f}s)")
        
        # 4. 保存
        task_save_dir = os.path.join(CKPT_ROOT, f"task_{task_id}")
        model.bert.save_pretrained(task_save_dir)
        torch.save(model.classifier.state_dict(), os.path.join(task_save_dir, "head.pth"))
        
        # 5. 评估
        print(f"📊 Evaluating Baseline 0 ~ {task_id}...")
        current_step_accs = []
        
        for eval_id in range(task_id + 1):
            test_file = os.path.join(DATA_ROOT, f"task_{eval_id}", "test.json")
            test_loader = DataLoader(NewsDataset(test_file, tokenizer), batch_size=32, shuffle=False)
            
            # 初始化评估模型
            eval_model = HOPBertClassifier(
                model_dir=model_path, 
                num_classes=NUM_CLASSES, 
                hop_order=HOP_ORDER, # <--- 关键点
                use_lora=False 
            )
            
            # 加载 LoRA
            adapter_path = os.path.join(CKPT_ROOT, f"task_{eval_id}")
            eval_model.bert = PeftModel.from_pretrained(eval_model.bert, adapter_path, is_trainable=False)
            
            # 加载 Head
            head_path = os.path.join(CKPT_ROOT, f"task_{eval_id}", "head.pth")
            eval_model.classifier.load_state_dict(torch.load(head_path))
            eval_model.to(device)
            
            acc = evaluate(eval_model, test_loader, device)
            current_step_accs.append(acc)
        
        # 记录结果 (确保转换为纯 float，避免 numpy 格式写入文件出错)
        step_avg = float(np.mean(current_step_accs))
        avg_accuracies.append(step_avg)
        print(f"👉 Baseline Step {task_id} Avg Acc: {step_avg*100:.2f}%")

    # === 保存 Baseline 结果 ===
    print("\n" + "="*40)
    print(f"🏆 Baseline Final Accuracy: {avg_accuracies[-1]*100:.2f}%")
    
    # 使用 JSON 格式保存，避免之前的读取错误
    with open(LOG_FILE, "w") as f:
        # 写入一个纯 Python 列表
        f.write(f"Avg Acc History: {json.dumps(avg_accuracies)}\n")
        
    print(f"📝 结果已保存至 {LOG_FILE}")

if __name__ == "__main__":
    main()