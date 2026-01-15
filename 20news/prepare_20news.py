import os
import json
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# 1. 设置保存路径
OUTPUT_DIR = "data/20news"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 正在下载 20Newsgroups 数据集 (DSW环境通常几秒钟)...")
# 这里我们移除 header/footer/quotes，防止模型通过元数据作弊，这是学术界标准做法
train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

all_labels = train_data.target_names
print(f"✅ 下载完成，共有 {len(all_labels)} 个类别。")

# 2. 定义 10 个任务 (每个任务 2 个类)
# 这是 Continual Learning 文献中标准的拆分方式
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
    """处理并保存数据"""
    for task_id, class_names in enumerate(TASK_SPLITS):
        # 创建任务目录
        task_dir = os.path.join(OUTPUT_DIR, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # 找到这两个类在原始数据中的 ID
        class_indices = [all_labels.index(c) for c in class_names]
        
        samples = []
        for text, label in zip(data_source.data, targets_source):
            if label in class_indices:
                # 过滤掉过短的垃圾文本
                if len(text.strip()) < 10:
                    continue
                
                # 将 label 映射为 0 或 1 (二分类)
                # 例如 Task 0 中: comp.graphics -> 0, comp.os... -> 1
                binary_label = class_indices.index(label)
                
                samples.append({
                    "sentence": text,
                    "label": binary_label
                })
        
        # 保存
        out_file = os.path.join(task_dir, f"{split_name}.json")
        with open(out_file, 'w') as f:
            json.dump(samples, f, indent=2)
            
        print(f"  Task {task_id} [{split_name}]: 生成 {len(samples)} 条样本 ({class_names})")

print("\n📦 开始划分任务...")
save_split(train_data, train_data.target, 'train')
save_split(test_data, test_data.target, 'test')
print(f"\n✅ 数据准备完毕！保存在: {OUTPUT_DIR}")