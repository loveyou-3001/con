import os
import json
import numpy as np
from datasets import load_dataset

# --- 1. 核心配置：使用国内镜像加速 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OUTPUT_DIR = "data/20news"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 正在通过 HuggingFace 镜像下载 20Newsgroups...")

# 加载 SetFit 提供的清洗版 20Newsgroups (包含 label_text 字段，方便处理)
# 这一步会走 hf-mirror.com，速度飞快
dataset = load_dataset("SetFit/20_newsgroups")

print("✅ 下载完成！")

# --- 2. 定义任务划分 (保持与之前一致) ---
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

def process_and_save(hf_split_name, local_split_name):
    """处理 HuggingFace 数据并保存为 TIL JSON 格式"""
    data_source = dataset[hf_split_name]
    
    # 预处理：按类别名称分组，加速查找
    from collections import defaultdict
    data_by_class = defaultdict(list)
    
    for item in data_source:
        text = item['text']
        label_name = item['label_text']
        # 过滤过短文本
        if len(text.strip()) > 10:
            data_by_class[label_name].append(text)
            
    # 按任务保存
    for task_id, class_names in enumerate(TASK_SPLITS):
        task_dir = os.path.join(OUTPUT_DIR, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        samples = []
        # class_names[0] -> label 0
        # class_names[1] -> label 1
        
        # 处理类别 0
        for text in data_by_class[class_names[0]]:
            samples.append({"sentence": text, "label": 0})
            
        # 处理类别 1
        for text in data_by_class[class_names[1]]:
            samples.append({"sentence": text, "label": 1})
            
        # 保存为 JSON
        out_file = os.path.join(task_dir, f"{local_split_name}.json")
        with open(out_file, 'w') as f:
            json.dump(samples, f, indent=2)
            
        print(f"  Task {task_id} [{local_split_name}]: 生成 {len(samples)} 条样本 ({class_names})")

print("\n📦 开始划分任务...")
process_and_save('train', 'train')
process_and_save('test', 'test')

print(f"\n✅ 数据准备完毕！保存在: {OUTPUT_DIR}")
print("👉 现在你可以直接运行 python run_20news.py 了")