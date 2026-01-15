import os
import shutil
import json
import random
import subprocess
from collections import Counter

# === 配置 ===
REPO_URL = "https://github.com/ZixuanKe/LifelongSentClass.git"
TEMP_DIR = "temp_dsc_repo_univ"
OUTPUT_ROOT = "data/dsc_small"
TRAIN_PER_CLASS = 100
SEED = 42

def save_json(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def parse_and_stat_file(file_path):
    """
    读取文件，返回 (数据列表, 标签统计)
    """
    data = []
    raw_labels = []
    
    if not os.path.exists(file_path):
        return data, Counter()
        
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 从右侧分割
            parts = line.rsplit('\t', 1)
            if len(parts) != 2:
                continue
                
            text, label_str = parts
            text = text.strip()
            label_str = label_str.strip()
            
            # 去除引号
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            
            # 记录原始标签用于统计
            raw_labels.append(label_str)
            
            # === 万能映射逻辑 ===
            label = None
            
            # 常见的 Positive 标记
            if label_str in ['1', '1.0', 'pos', 'positive', '5', '4']:
                label = 1
            # 常见的 Negative 标记 (增加了 -1)
            elif label_str in ['0', '0.0', 'neg', 'negative', '-1', '-1.0', '1', '2']: 
                # 注意：有些数据集用 1=Neg, 2=Pos。
                # 但根据你之前的 Debug，Line 0 是 "highly recommend" 且 label=1
                # 所以 1 肯定是 Positive。
                # 那么 Negative 很可能是 -1 或 0。
                if label_str in ['-1', '-1.0', '0', '0.0', 'neg', 'negative']:
                    label = 0
            
            # 如果还没匹配上，暂时存为 None，后续在主函数里看统计
            if label is not None:
                data.append({'text': text, 'label': label})
                
    return data, Counter(raw_labels)

def main():
    print(f"🚀 [Universal] 准备 DSC (Small) 数据 - 自适应标签模式")
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print("📥 Cloning repository...")
    try:
        subprocess.check_call(f"git clone {REPO_URL} {TEMP_DIR}", shell=True)
    except Exception as e:
        print(f"❌ Git clone 失败: {e}")
        return

    raw_root = os.path.join(TEMP_DIR, "dat", "dsc")
    if not os.path.exists(raw_root):
        raw_root = os.path.join(TEMP_DIR, "data", "dsc")
    
    # 获取任务列表
    all_files = os.listdir(raw_root)
    domains = sorted(list(set([f.replace('_train.tsv', '') for f in all_files if f.endswith('_train.tsv')])))
    print(f"📂 发现 {len(domains)} 个领域。")
    
    random.seed(SEED)
    processed_count = 0
    
    for task_id, domain in enumerate(domains):
        print(f"\n📦 分析领域: {domain}")
        
        # 1. 读取所有 split 并统计原始标签
        full_pool = []
        total_stats = Counter()
        
        for split in ['train', 'test', 'dev']:
            fpath = os.path.join(raw_root, f"{domain}_{split}.tsv")
            parsed_data, stats = parse_and_stat_file(fpath)
            full_pool.extend(parsed_data)
            total_stats.update(stats)
            
        print(f"   📊 原始标签分布: {dict(total_stats)}")
        
        # 2. 检查是否因为映射失败导致丢失数据
        # 如果 total_stats 里有大量数据，但 full_pool 很小，说明映射规则漏了
        mapped_count = len(full_pool)
        raw_count = sum(total_stats.values())
        
        if mapped_count < raw_count:
            print(f"   ⚠️ 警告: 只有 {mapped_count}/{raw_count} 条数据被成功映射。请检查上方原始标签分布！")
            # 如果发现 '-1' 没被映射，这里会自动提示
        
        # 3. 分离正负样本
        pos_samples = [x for x in full_pool if x['label'] == 1]
        neg_samples = [x for x in full_pool if x['label'] == 0]
        
        print(f"   ✅ 有效样本: Pos={len(pos_samples)}, Neg={len(neg_samples)}")
        
        if len(pos_samples) < TRAIN_PER_CLASS or len(neg_samples) < TRAIN_PER_CLASS:
            print(f"   ❌ 样本依然不足，跳过。")
            continue

        # 4. 采样与保存
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        
        train_data = pos_samples[:TRAIN_PER_CLASS] + neg_samples[:TRAIN_PER_CLASS]
        random.shuffle(train_data)
        
        test_pos = pos_samples[TRAIN_PER_CLASS:]
        test_neg = neg_samples[TRAIN_PER_CLASS:]
        
        # 限制测试集大小
        if len(test_pos) > 250: test_pos = test_pos[:250]
        if len(test_neg) > 250: test_neg = test_neg[:250]
        
        test_data = test_pos + test_neg
        random.shuffle(test_data)
        
        final_task_id = processed_count
        save_json(train_data, os.path.join(OUTPUT_ROOT, f"task_{final_task_id}", "train.json"))
        save_json(test_data, os.path.join(OUTPUT_ROOT, f"task_{final_task_id}", "test.json"))
        
        print(f"   💾 保存 Task {final_task_id}")
        processed_count += 1
        if processed_count >= 10:
            break

    shutil.rmtree(TEMP_DIR)
    print(f"\n🎉 数据准备完成！路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()