import os
import json
import random
import pandas as pd
# 替换 datasets 为 modelscope.msdatasets
from modelscope.msdatasets import MsDataset
from tqdm import tqdm

# === 1. 配置 ===
# 论文配置: 10个任务(产品), 2分类
# Small设置: Train=100pos+100neg, Test=250pos+250neg
NUM_TASKS = 10
TRAIN_SAMPLES_PER_CLASS = 100 
TEST_SAMPLES_PER_CLASS = 250
SEED = 42
OUTPUT_ROOT = "data/dsc_small"

# 为了模拟10个不同领域，我们从 Amazon Reviews 中选取10个主要的大类
# 注意：实际论文用了特定10个产品，这里用10个大类作为最佳近似
TARGET_CATEGORIES = [
    'book', 'digital_ebook_purchase', 'wireless', 'pc', 'home', 
    'apparel', 'beauty', 'drugstore', 'sports', 'toy'
]

def save_json(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def main():
    print(f"🚀 [ModelScope] 正在准备 DSC (Small) 数据集...")
    print(f"   Train: {TRAIN_SAMPLES_PER_CLASS*2} samples/task (Few-shot!)")
    print(f"   Test : {TEST_SAMPLES_PER_CLASS*2} samples/task")

    try:
        # === 核心修改：使用 ModelScope 下载 ===
        # 阿里云内网加速，无需 HF_ENDPOINT
        # subset_name='en' 表示下载英文版
        print("📥 正在从 ModelScope 镜像下载 'amazon_reviews_multi' (subset='en')...")
        ds = MsDataset.load('amazon_reviews_multi', subset_name='en', split='train')
        
        # 转换为 Pandas DataFrame
        # ModelScope 的 dataset 可能是个生成器或 list，需要转一下
        print("🔄 正在转换为 DataFrame...")
        # ds.to_pandas() 在某些版本可能不可用，使用标准方法转换
        # MsDataset 通常表现为 HuggingFace dataset 的 wrapper
        if hasattr(ds, 'to_pandas'):
            df = ds.to_pandas()
        else:
            # 如果没有直接的方法，则手动转换
            data_list = [item for item in tqdm(ds, desc="Loading data")]
            df = pd.DataFrame(data_list)
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 建议：请检查 modelscope 库是否安装 (pip install modelscope)")
        return

    # === 2. 数据清洗与分桶 (逻辑保持不变) ===
    print(f"🧹 数据加载成功 (共 {len(df)} 条)，开始清洗...")
    
    # 评分 1-2 为负(0), 4-5 为正(1), 3分丢弃
    df = df[df['stars'] != 3]
    df['label'] = df['stars'].apply(lambda x: 1 if x > 3 else 0)
    
    random.seed(SEED)
    
    # 遍历10个任务类别
    for task_id, category in enumerate(TARGET_CATEGORIES):
        print(f"\n📦 处理 Task {task_id}: Domain = {category}")
        
        # 筛选当前领域的数据
        domain_df = df[df['product_category'] == category]
        
        # 分离正负样本
        pos_df = domain_df[domain_df['label'] == 1]
        neg_df = domain_df[domain_df['label'] == 0]
        
        # 检查数据够不够
        if len(pos_df) < (TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS):
            print(f"⚠️ 警告: 类别 {category} 正样本不足，跳过")
            continue
            
        # === 3. 采样 (核心步骤) ===
        # 随机抽取 Train (100 pos + 100 neg)
        train_pos = pos_df.sample(n=TRAIN_SAMPLES_PER_CLASS, random_state=SEED)
        train_neg = neg_df.sample(n=TRAIN_SAMPLES_PER_CLASS, random_state=SEED)
        
        # 剩下的数据里抽取 Test (250 pos + 250 neg)
        remaining_pos = pos_df.drop(train_pos.index)
        remaining_neg = neg_df.drop(train_neg.index)
        
        test_pos = remaining_pos.sample(n=TEST_SAMPLES_PER_CLASS, random_state=SEED)
        test_neg = remaining_neg.sample(n=TEST_SAMPLES_PER_CLASS, random_state=SEED)
        
        # 合并并打乱
        train_data = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=SEED)
        test_data = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=SEED)
        
        # === 4. 格式化并保存 ===
        # 转换为NewsDataset需要的格式: {'text': ..., 'label': ...}
        train_list = []
        for _, row in train_data.iterrows():
            train_list.append({
                'text': f"{row['review_title']} . {row['review_body']}", # 拼接标题和内容
                'label': int(row['label'])
            })
            
        test_list = []
        for _, row in test_data.iterrows():
            test_list.append({
                'text': f"{row['review_title']} . {row['review_body']}",
                'label': int(row['label'])
            })
            
        save_path_train = os.path.join(OUTPUT_ROOT, f"task_{task_id}", "train.json")
        save_path_test = os.path.join(OUTPUT_ROOT, f"task_{task_id}", "test.json")
        
        save_json(train_list, save_path_train)
        save_json(test_list, save_path_test)
        
        print(f"   ✅ 已生成: Train({len(train_list)}), Test({len(test_list)})")

    print(f"\n🎉 DSC (Small) 数据准备完成！路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()