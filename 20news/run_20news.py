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

# 引入你的 HOP 模型和数据集定义
# 确保 hop_model.py 和 news_dataset.py 在同一目录下
from hop_model import HOPBertClassifier, get_bert_path
from news_dataset import NewsDataset
from peft import PeftModel

# === 全局配置 ===
DATA_ROOT = "data/20news"           # 数据路径
CKPT_ROOT = "checkpoints_20news"    # 权重保存路径
NUM_TASKS = 10                      # 任务总数
NUM_CLASSES = 2                     # 二分类
HOP_ORDER = 3                       # 3阶 HOP
EPOCHS = 3                          # 训练轮数
BATCH_SIZE = 16                     # 批次大小
LR = 5e-4                           # 学习率 (LoRA通常大一点)

# 忽略一些不影响运行的警告
warnings.filterwarnings("ignore", category=FutureWarning)

def train_one_task(model, loader, device):
    """训练单个任务"""
    # 过滤出需要更新的参数 (LoRA参数 + Head参数)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        # 使用 tqdm 显示进度条
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
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

def evaluate(model, loader, device):
    """评估模型"""
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

def verify_lora_loaded(model, task_id):
    """调试函数：检查 LoRA 权重是否非零且已加载"""
    try:
        # 寻找第一个 LoRA A 矩阵
        for name, param in model.bert.named_parameters():
            if "lora_A" in name:
                # 获取第一个数值
                val = param.data.reshape(-1)[0].item()
                print(f"    🔍 [Debug] Task {task_id} Adapter Checksum (First Val): {val:.6f}")
                if val == 0.0:
                    print("    ⚠️  警告: LoRA 权重似乎是全 0 (可能是初始化问题)！")
                return
        print("    ⚠️  警告: 未在模型中找到名为 'lora_A' 的参数！")
    except Exception as e:
        print(f"    ⚠️  验证出错: {e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # 确保保存目录存在
    os.makedirs(CKPT_ROOT, exist_ok=True)
    
    # 获取 BERT 路径 (自动处理 ModelScope/HF)
    model_path = get_bert_path()
    print(f"📂 BERT Base Path: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 记录历史数据用于绘图
    # 格式: accuracy_matrix[t_learned][t_eval]
    accuracy_matrix = np.zeros((NUM_TASKS, NUM_TASKS))
    avg_accuracies = []

    print(f"\n🔥 开始 20Newsgroups (10 Tasks) 持续学习实验...")

    # === 主循环：依次学习 10 个任务 ===
    for task_id in range(NUM_TASKS):
        print(f"\n{'='*20} Current Task: {task_id} / {NUM_TASKS-1} {'='*20}")
        
        # --- 1. 准备训练数据 ---
        train_file = os.path.join(DATA_ROOT, f"task_{task_id}", "train.json")
        train_loader = DataLoader(NewsDataset(train_file, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        
        # --- 2. 初始化训练模型 ---
        # 训练时 use_lora=True，让 HOPBertClassifier 自动创建一个随机初始化的 Adapter
        print("🛠️  Initializing Training Model...")
        model = HOPBertClassifier(
            model_dir=model_path, 
            num_classes=NUM_CLASSES, 
            hop_order=HOP_ORDER, 
            use_lora=True
        ).to(device)
        
        # --- 3. 训练 ---
        start_time = time.time()
        train_one_task(model, train_loader, device)
        print(f"✅ Task {task_id} Trained. Time: {time.time()-start_time:.1f}s")
        
        # --- 4. 保存模型 ---
        task_save_dir = os.path.join(CKPT_ROOT, f"task_{task_id}")
        
        # 保存 LoRA 部分 (PeftModel 提供了 save_pretrained)
        model.bert.save_pretrained(task_save_dir)
        
        # 保存 Classifier Head (HOP池化层没有参数，只需保存 MLP)
        torch.save(model.classifier.state_dict(), os.path.join(task_save_dir, "head.pth"))
        print(f"💾 Checkpoint saved to: {task_save_dir}")
        
        # --- 5. 评估循环 (Evaluation Loop) ---
        # 验证抗遗忘能力：在当前时刻，测试所有已经学过的任务 (0 ~ task_id)
        print(f"📊 Evaluating Task 0 to {task_id}...")
        
        current_step_accs = []
        
        for eval_id in range(task_id + 1):
            test_file = os.path.join(DATA_ROOT, f"task_{eval_id}", "test.json")
            test_loader = DataLoader(NewsDataset(test_file, tokenizer), batch_size=32, shuffle=False)
            
            # [关键修复] 初始化评估模型
            # use_lora=False: 先加载纯 BERT，不要自带随机 Adapter
            eval_model = HOPBertClassifier(
                model_dir=model_path, 
                num_classes=NUM_CLASSES, 
                hop_order=HOP_ORDER, 
                use_lora=False 
            )
            
            # [关键修复] 使用 from_pretrained 挂载特定任务的 Adapter
            adapter_path = os.path.join(CKPT_ROOT, f"task_{eval_id}")
            eval_model.bert = PeftModel.from_pretrained(
                eval_model.bert, 
                adapter_path,
                is_trainable=False
            )
            
            # 验证权重是否正确加载
            if eval_id == 0: # 只打印第一个任务的check作为代表，避免刷屏
                verify_lora_loaded(eval_model, eval_id)

            # 加载对应的分类头
            head_path = os.path.join(CKPT_ROOT, f"task_{eval_id}", "head.pth")
            eval_model.classifier.load_state_dict(torch.load(head_path))
            
            eval_model.to(device)
            
            # 推理
            acc = evaluate(eval_model, test_loader, device)
            accuracy_matrix[task_id, eval_id] = acc
            current_step_accs.append(acc)
            # print(f"    Task {eval_id} Acc: {acc*100:.2f}%")
        
        # 计算当前 Step 的平均准确率
        step_avg = np.mean(current_step_accs)
        avg_accuracies.append(step_avg)
        print(f"👉 Step {task_id} Finished | Avg Acc: {step_avg*100:.2f}%")

    # === 实验结束总结 ===
    print("\n" + "="*40)
    print("🎉 All Tasks Completed!")
    print(f"🏆 Final Average Accuracy: {avg_accuracies[-1]*100:.2f}%")
    
    # 计算遗忘率 (Forgetting Measure)
    # Task 0 在 Step 0 的准确率 vs 在 Step 9 的准确率
    if NUM_TASKS > 1:
        acc_initial = accuracy_matrix[0, 0]
        acc_final = accuracy_matrix[NUM_TASKS-1, 0]
        forgetting = acc_initial - acc_final
        print(f"📉 Forgetting on Task 0: {forgetting*100:.2f}% (Lower is better)")
        print(f"   (Initial: {acc_initial*100:.2f}% -> Final: {acc_final*100:.2f}%)")

    # 保存结果到文件以便画图
    with open("results_log.txt", "w") as f:
        f.write(f"Avg Acc History: {avg_accuracies}\n")
        f.write(f"Full Matrix:\n{accuracy_matrix}\n")
    print("📝 Results saved to results_log.txt")

if __name__ == "__main__":
    main()