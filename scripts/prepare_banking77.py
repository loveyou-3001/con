import os
import json
import random
from datasets import load_dataset # 需要 pip install datasets

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def prepare_banking77(output_root="data/banking77", num_tasks=7):
    print("📥 Downloading Banking77 dataset via HuggingFace...")
    # 加载 Banking77 数据集
    dataset = load_dataset("mteb/banking77", trust_remote_code=True)
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    # 获取所有标签名称
    # Banking77 的 label 是整数，我们需要知道总共有多少类
    # 这里我们直接用 label 字段
    all_labels = sorted(list(set(train_data['label'])))
    num_classes = len(all_labels)
    print(f"📊 Total classes: {num_classes}")
    
    # 随机打乱类别顺序以构建持续学习任务
    random.seed(42)
    random.shuffle(all_labels)
    
    # 切分任务
    classes_per_task = num_classes // num_tasks
    remaining = num_classes % num_tasks
    
    print(f"🔄 Splitting into {num_tasks} tasks...")
    
    current_idx = 0
    for task_id in range(num_tasks):
        # 计算当前任务包含的类别
        n_cls = classes_per_task + (1 if task_id < remaining else 0)
        task_labels = set(all_labels[current_idx : current_idx + n_cls])
        current_idx += n_cls
        
        # 筛选数据
        task_train = []
        task_test = []
        
        for item in train_data:
            if item['label'] in task_labels:
                task_train.append({"text": item['text'], "label": item['label']})
                
        for item in test_data:
            if item['label'] in task_labels:
                task_test.append({"text": item['text'], "label": item['label']})
        
        # 保存
        task_dir = os.path.join(output_root, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        save_jsonl(task_train, os.path.join(task_dir, "train.json"))
        save_jsonl(task_test, os.path.join(task_dir, "test.json"))
        
        print(f"  ✅ Task {task_id}: {len(task_labels)} classes | Train: {len(task_train)} | Test: {len(task_test)}")

if __name__ == "__main__":
    prepare_banking77()