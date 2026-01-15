import os
import shutil
import json
import random
import glob
import subprocess

# === 配置 ===
REPO_URL = "https://github.com/ZixuanKe/LifelongSentClass.git"
TEMP_DIR = "temp_dsc_repo"
OUTPUT_ROOT = "data/dsc_small"

# DSC Small 设置: 每个任务 Train=200 (100+100), Test=全部剩余或固定500
# 论文中 Test 通常是 250+250，这里我们取剩余的全部作为 Test 以最大化评估
TRAIN_SAMPLES_PER_CLASS = 100
SEED = 42

def run_cmd(cmd):
    print(f"执行: {cmd}")
    subprocess.check_call(cmd, shell=True)

def save_json(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def parse_review_file(file_path):
    """
    解析原始数据文件。
    ZixuanKe 的数据格式通常是 XML-like 或者 line-based text。
    根据 dat/dsc 中的文件，通常格式是：
    label \t review_text
    或者
    <review>...</review>
    我们需要根据实际内容自适应。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 尝试拆分 Label 和 Text
        # 常见格式: "positive <tab> this is a good book"
        # 或者: "1 <tab> review..."
        parts = line.split('\t', 1)
        if len(parts) == 2:
            label_str, text = parts
            
            # 转换标签为 0/1
            # 假设: 1/pos/positive -> 1, 0/neg/negative -> 0
            label_str = label_str.lower()
            if label_str in ['positive', 'pos', '1', '5.0', '4.0']:
                label = 1
            elif label_str in ['negative', 'neg', '0', '1.0', '2.0']:
                label = 0
            else:
                continue # 跳过中性或无法解析的
            
            data.append({'text': text, 'label': label})
            
    return data

def main():
    print(f"🚀 开始准备 DSC (Small) 数据 (Source: GitHub {REPO_URL})")
    
    # 1. Clone 仓库
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print("📥 Cloning repository... (可能需要几秒钟)")
    try:
        run_cmd(f"git clone {REPO_URL} {TEMP_DIR}")
    except Exception as e:
        print(f"❌ Git clone 失败: {e}")
        return

    # 2. 查找 DSC 数据文件
    # 路径通常是 temp/dat/dsc/ 下的 10 个文件
    raw_data_path = os.path.join(TEMP_DIR, "dat", "dsc")
    if not os.path.exists(raw_data_path):
        # 备用路径尝试
        raw_data_path = os.path.join(TEMP_DIR, "data", "dsc")
    
    if not os.path.exists(raw_data_path):
        print(f"❌ 找不到数据目录: {raw_data_path}")
        print(f"目录结构: {os.listdir(TEMP_DIR)}")
        return

    # 获取所有文件 (排除 readme 等)
    files = sorted([f for f in os.listdir(raw_data_path) if not f.startswith('.')])
    print(f"📂 发现 {len(files)} 个数据文件: {files}")
    
    if len(files) == 0:
        print("❌ 目录下没有文件！")
        return

    random.seed(SEED)

    # 3. 处理每个文件为一个 Task
    for task_id, filename in enumerate(files):
        print(f"\n📦 Processing Task {task_id}: {filename}")
        
        file_path = os.path.join(raw_data_path, filename)
        all_samples = parse_review_file(file_path)
        
        print(f"   解析出 {len(all_samples)} 条样本")
        if len(all_samples) < 200:
            print("   ⚠️ 样本太少，跳过")
            continue

        # 分离正负样本
        pos_samples = [x for x in all_samples if x['label'] == 1]
        neg_samples = [x for x in all_samples if x['label'] == 0]
        
        # 采样 Train (Small Setting)
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        
        train_pos = pos_samples[:TRAIN_SAMPLES_PER_CLASS]
        train_neg = neg_samples[:TRAIN_SAMPLES_PER_CLASS]
        
        # 剩余作为 Test
        test_pos = pos_samples[TRAIN_SAMPLES_PER_CLASS:]
        test_neg = neg_samples[TRAIN_SAMPLES_PER_CLASS:]
        
        # 如果测试集太大，可以截断 (论文通常用 250+250)
        # test_pos = test_pos[:250]
        # test_neg = test_neg[:250]
        
        train_data = train_pos + train_neg
        test_data = test_pos + test_neg
        
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        # 保存
        save_path_train = os.path.join(OUTPUT_ROOT, f"task_{task_id}", "train.json")
        save_path_test = os.path.join(OUTPUT_ROOT, f"task_{task_id}", "test.json")
        
        save_json(train_data, save_path_train)
        save_json(test_data, save_path_test)
        
        print(f"   ✅ Task {task_id} Ready! Train: {len(train_data)}, Test: {len(test_data)}")

    # 4. 清理
    shutil.rmtree(TEMP_DIR)
    print(f"\n🎉 所有数据处理完成！路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()