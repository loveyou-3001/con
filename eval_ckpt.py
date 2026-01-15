import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score

# 复用你项目中的模块
from src.model import HOPBertClassifier, get_bert_path
from src.dataset import JSONLDataset

def parse_args():
    parser = argparse.ArgumentParser(description="手动加载 Checkpoint 进行评估")
    
    # 核心参数：指向你要评估的那个 task_x 文件夹
    parser.add_argument("--ckpt_dir", type=str, required=True, 
                        help="Checkpint 文件夹路径, 例如: output/clinc_hop_prototype_balanced/task_12")
    
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hop_order", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def load_model(ckpt_dir, num_classes, hop_order, device):
    print(f"🔄 正在加载模型权重: {ckpt_dir}")
    
    # 1. 获取基础 BERT 路径
    model_path = get_bert_path()
    
    # 2. 初始化空模型 (use_lora=False, 因为我们要手动加载 PeftModel)
    model = HOPBertClassifier(model_path, num_classes=num_classes, hop_order=hop_order, use_lora=False)
    
    # 3. 加载 LoRA 权重 (Adapter)
    # 这会将 task_x 里的 adapter_model.bin 加载进去
    model.bert = PeftModel.from_pretrained(model.bert, ckpt_dir, is_trainable=False)
    
    # 4. 加载分类头 (Head)
    head_path = os.path.join(ckpt_dir, "head.pth")
    if os.path.exists(head_path):
        model.classifier.load_state_dict(torch.load(head_path, map_location=device))
        print("✅ Classifier Head 加载成功")
    else:
        print(f"❌ 警告: 找不到 {head_path}，分类头将使用随机初始化！")
    
    model.to(device)
    model.eval()
    return model

def evaluate_one_task(model, tokenizer, data_root, task_id, batch_size, device):
    test_path = os.path.join(data_root, f"task_{task_id}", "test.json")
    if not os.path.exists(test_path):
        return None
    
    dataset = JSONLDataset(test_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
            
    return accuracy_score(gts, preds)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从 ckpt_dir 路径中解析出当前的 task_id (例如 task_12 -> 12)
    # 这样我们知道应该评估 Task 0 到 Task 12
    try:
        folder_name = os.path.basename(os.path.normpath(args.ckpt_dir))
        current_task_id = int(folder_name.split('_')[1])
    except:
        print("⚠️ 无法从路径解析 Task ID，默认评估 Task 0-14")
        current_task_id = 14

    # 加载模型
    model = load_model(args.ckpt_dir, args.num_classes, args.hop_order, device)
    tokenizer = AutoTokenizer.from_pretrained(get_bert_path())
    
    print(f"\n🧐 开始评估 Task 0 -> Task {current_task_id} ...")
    print("-" * 40)
    
    accs = []
    for task_id in range(current_task_id + 1):
        acc = evaluate_one_task(model, tokenizer, args.data_root, task_id, args.batch_size, device)
        if acc is not None:
            accs.append(acc)
            print(f"Task {task_id}: {acc*100:.2f}%")
        else:
            print(f"Task {task_id}: 数据文件不存在")
            
    print("-" * 40)
    avg_acc = np.mean(accs) * 100
    print(f"🏆 Average Accuracy: {avg_acc:.2f}%")
    print(f"📊 Matrix Row: {[f'{x*100:.1f}' for x in accs]}")

    # 特别检查 Task 0 (遗忘程度)
    if len(accs) > 0:
        print(f"🧠 Task 0 Retention: {accs[0]*100:.2f}%")

if __name__ == "__main__":
    main()