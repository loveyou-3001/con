import os
import shutil
import json
import random
import subprocess

# === 配置 ===
REPO_URL = "https://github.com/ZixuanKe/LifelongSentClass.git"
TEMP_DIR = "temp_dsc_repo_final"
OUTPUT_ROOT = "data/dsc_small"

# DSC Small (Few-shot) 设置
TRAIN_SAMPLES_PER_CLASS = 100  # 100 pos + 100 neg
SEED = 42

def save_json(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def parse_review_file(file_path):
    """
    针对 debug 发现的格式进行解析:
    Text \t Label
    """
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 按最后一个 tab 分割，防止文本内部有 tab
        parts = line.rsplit('\t', 1)
        
        if len(parts) == 2:
            text, label_str = parts
            text = text.strip()
            label_str = label_str.strip()
            
            # 去除包裹的引号 (根据你的 Debug 输出)
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]

            # 转换标签
            if label_str == '1':
                label = 1
            elif label_str == '0':
                label = 0
            else:
                continue # 跳过异常行
            
            data.append({'text': text, 'label': label})
            
    return data

def main():
    print(f"🚀 [Final Fix] 准备 DSC (Small) 数据...")
    
    # 1. Clone
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print("📥 Cloning repository...")
    try:
        subprocess.check_call(f"git clone {REPO_URL} {TEMP_DIR}", shell=True)
    except Exception as e:
        print(f"❌ Git clone 失败: {e}")
        return

    # 2. 定位数据目录
    raw_data_path = os.path.join(TEMP_DIR, "dat", "dsc")
    if not os.path.exists(raw_data_path):
        raw_data_path = os.path.join(TEMP_DIR, "data", "dsc") # 备用检查
    
    # 获取文件列表 (确保顺序一致)
    files = sorted([f for f in os.listdir(raw_data_path) if f.endswith('train.tsv')])
    print(f"📂 找到 {len(files)} 个训练文件 (每个文件代表一个 Task)")

    random.seed(SEED)

    # 3. 处理
    for task_id, train_filename in enumerate(files):
        print(f"\n📦 Task {task_id}: {train_filename}")
        
        # 这里有个坑：作者仓库里 train/test/dev 是分开的文件
        # 例如: Amazon_Instant_Video_train.tsv
        # 我们需要同时读取对应的 test.tsv
        
        prefix = train_filename.replace('_train.tsv', '')
        test_filename = f"{prefix}_test.tsv"
        
        train_path = os.path.join(raw_data_path, train_filename)
        test_path = os.path.join(raw_data_path, test_filename)
        
        # 解析
        full_train_data = parse_review_file(train_path)
        full_test_data = parse_review_file(test_path) if os.path.exists(test_path) else []
        
        if not full_train_data:
            print(f"   ⚠️ 警告: {train_filename} 解析为空，检查格式！")
            continue

        # === 构造 Small Setting (Few-shot) ===
        # 论文设置: 挑选 100 pos + 100 neg 作为训练
        # 剩余的 train数据 + 所有的 test数据 -> 都可以作为测试集
        # (或者严格遵循论文，只取test数据作为测试)
        
        pos_samples = [x for x in full_train_data if x['label'] == 1]
        neg_samples = [x for x in full_train_data if x['label'] == 0]
        
        if len(pos_samples) < 100 or len(neg_samples) < 100:
            print(f"   ⚠️ 样本不足 200，跳过 Task {task_id}")
            continue

        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        
        # 选取 Few-shot 训练集
        final_train = pos_samples[:100] + neg_samples[:100]
        random.shuffle(final_train)
        
        # 构建测试集 (使用原始的 test.tsv)
        # 如果 test.tsv 没有或太少，可以用剩余的 train 补充，但优先用 test 文件
        final_test = full_test_data
        if len(final_test) > 1000:
            final_test = random.sample(final_test, 1000) # 限制测试集大小加速评估
            
        save_path_train = os.path.join(OUTPUT_ROOT, f"task_{task_id}", "train.json")
        save_path_test = os.path.join(OUTPUT_ROOT, f"task_{task_id}", "test.json")
        
        save_json(final_train, save_path_train)
        save_json(final_test, save_path_test)
        
        print(f"   ✅ 生成完毕: Train({len(final_train)}), Test({len(final_test)})")

    # 4. 清理
    shutil.rmtree(TEMP_DIR)
    print(f"\n🎉 数据准备完成！路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()