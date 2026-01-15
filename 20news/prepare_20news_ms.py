import os
import json
# 1. 使用 ModelScope 数据集接口
from modelscope.msdatasets import MsDataset

OUTPUT_DIR = "data/20news"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 正在从 ModelScope 下载 20Newsgroups...")

# ModelScope 上的 20news 数据集 ID 通常为 'news20'
# splitting 默认通常有 'train' 和 'test'
ds_train = MsDataset.load('news20', subset_name='default', split='train')
ds_test = MsDataset.load('news20', subset_name='default', split='test')

print(f"✅ 下载完成！训练集: {len(ds_train)}, 测试集: {len(ds_test)}")

# ModelScope 的 label 是数字 (0-19)，我们需要手动映射一下对应的类名
# 20Newsgroups 标准顺序
LABELS = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
    'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 
    'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 
    'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

# 任务划分配置 (同前)
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

def process_and_save(dataset_obj, split_name):
    # 预处理：按 Label ID 分组
    from collections import defaultdict
    data_by_label_id = defaultdict(list)
    
    for item in dataset_obj:
        # ModelScope 返回的 item key 可能是 'text' 和 'label'
        text = item['text']
        label_id = item['label']
        if len(text.strip()) > 10:
            data_by_label_id[label_id].append(text)
            
    # 按任务保存
    for task_id, class_names in enumerate(TASK_SPLITS):
        task_dir = os.path.join(OUTPUT_DIR, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # 找到这俩类名对应的 ID
        # 注意：这里需要根据 LABELS 列表反查 ID
        target_ids = [LABELS.index(name) for name in class_names]
        
        samples = []
        # 处理第一个类 -> 存为 label 0
        for text in data_by_label_id[target_ids[0]]:
            samples.append({"sentence": text, "label": 0})
            
        # 处理第二个类 -> 存为 label 1
        for text in data_by_label_id[target_ids[1]]:
            samples.append({"sentence": text, "label": 1})
            
        out_file = os.path.join(task_dir, f"{split_name}.json")
        with open(out_file, 'w') as f:
            json.dump(samples, f, indent=2)
            
        print(f"  Task {task_id} [{split_name}]: 生成 {len(samples)} 条样本")

print("\n📦 开始划分任务...")
process_and_save(ds_train, 'train')
process_and_save(ds_test, 'test')
print("\n✅ 处理完成！")