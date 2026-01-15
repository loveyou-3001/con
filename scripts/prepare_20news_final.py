# 文件路径: scripts/prepare_20news_final.py
import os
import json
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# 1. 设置保存路径 (适配 main.py 的 --data_root)
OUTPUT_DIR = "data/20news"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 正在下载/加载 20Newsgroups 数据集...")
# 移除 header/footer/quotes 以防止信息泄露 (Standard CL Setup)
train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

all_labels = train_data.target_names
print(f"✅ 数据就绪，共有 {len(all_labels)} 个类别。")

# 2. 定义 10 个任务 (每任务 2 类)
TASK_SPLITS = [
    ['comp.graphics', 'comp.os.ms-windows.misc'],           # Task 0
    ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'],  # Task 1
    ['comp.windows.x', 'misc.forsale'],                     # Task 2
    ['rec.autos', 'rec.motorcycles'],                       # Task 3
    ['rec.sport.baseball', 'rec.sport.hockey'],             # Task 4
    ['sci.crypt', 'sci.electronics'],                       # Task 5
    ['sci.med', 'sci.space'],                               # Task 6
    ['soc.religion.christian', 'talk.politics.guns'],       # Task 7
    ['talk.politics.mideast', 'talk.politics.misc'],        # Task 8
    ['alt.atheism', 'talk.religion.misc']                   # Task 9
]

def save_split(data_source, targets_source, split_name):
    """处理并保存数据为 JSONL 格式 (适配 main.py)"""
    for task_id, class_names in enumerate(TASK_SPLITS):
        task_dir = os.path.join(OUTPUT_DIR, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # 获取类别索引
        class_indices = [all_labels.index(c) for c in class_names]
        
        samples = []
        for text, label in zip(data_source.data, targets_source):
            if label in class_indices:
                # 过滤极短文本 (清洗)
                if len(text.strip()) < 10:
                    continue
                
                # 映射为 0/1 二分类
                binary_label = class_indices.index(label)
                
                # 🔥 修正 1: 键名改为 'text'
                samples.append({
                    "text": text.replace("\n", " ").strip(), # 去除换行符，保证一行一条
                    "label": binary_label
                })
        
        # 🔥 修正 2: 保存为 JSONL 格式 (Line-delimited JSON)
        out_file = os.path.join(task_dir, f"{split_name}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            
        print(f"  Task {task_id} [{split_name}]: 生成 {len(samples)} 条样本")

print("\n📦 开始划分任务并保存为 JSONL...")
save_split(train_data, train_data.target, 'train')
save_split(test_data, test_data.target, 'test')
print(f"\n✅ 20News 数据准备完毕！路径: {OUTPUT_DIR}")