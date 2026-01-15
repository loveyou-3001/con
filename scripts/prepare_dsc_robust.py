import os
import shutil
import json
import random
import subprocess
import glob

# === 配置 ===
REPO_URL = "https://github.com/ZixuanKe/LifelongSentClass.git"
TEMP_DIR = "temp_dsc_repo_robust"
OUTPUT_ROOT = "data/dsc_small"

# Small Setting: Train = 100 pos + 100 neg
TRAIN_PER_CLASS = 100
SEED = 42

def save_json(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def parse_file(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
        
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 尝试从右边分割 Label (Text \t Label)
            parts = line.rsplit('\t', 1)
            if len(parts) != 2:
                continue
                
            text, label_str = parts
            text = text.strip()
            label_str = label_str.strip().lower()
            
            # 去除可能存在的引号
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            
            # 标签归一化
            label = None
            if label_str in ['1', '1.0', 'pos', 'positive']:
                label = 1
            elif label_str in ['0', '0.0', 'neg', 'negative']:
                label = 0
            
            if label is not None:
                data.append({'text': text, 'label': label})
                
    return data

def main():
    print(f"🚀 [Robust] 准备 DSC (Small) 数据 - 全量合并采样模式")
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print("📥 Cloning repository...")
    try:
        subprocess.check_call(f"git clone {REPO_URL} {TEMP_DIR}", shell=True)
    except Exception as e:
        print(f"❌ Git clone 失败: {e}")
        return

    # 寻找数据根目录
    raw_root = os.path.join(TEMP_DIR, "dat", "dsc")
    if not os.path.exists(raw_root):
        raw_root = os.path.join(TEMP_DIR, "data", "dsc")
    
    # 获取所有任务名 (去重)
    # 文件名格式通常是: Domain_train.tsv
    all_files = os.listdir(raw_root)
    domains = set()
    for f in all_files:
        if f.endswith('_train.tsv'):
            domains.add(f.replace('_train.tsv', ''))
    
    domains = sorted(list(domains))
    print(f"📂 发现 {len(domains)} 个领域: {domains[:3]} ...")
    
    random.seed(SEED)
    
    # 论文只用了10个领域，如果发现更多，我们只取前10个或者指定的10个
    # 为了复现准确，我们按字母顺序取前10个，或者取特定的
    # 这里我们处理所有发现的领域，然后在 run_dsc.py 里限制 num_tasks
    
    processed_count = 0
    
    for task_id, domain in enumerate(domains):
        print(f"\n📦 处理领域: {domain}")
        
        # 1. 合并 Train + Test + Dev 所有数据
        pool = []
        for split in ['train', 'test', 'dev']:
            fpath = os.path.join(raw_root, f"{domain}_{split}.tsv")
            parsed = parse_file(fpath)
            pool.extend(parsed)
            # print(f"   - 加载 {split}: {len(parsed)} 条")
            
        # 2. 统计分布
        pos_samples = [x for x in pool if x['label'] == 1]
        neg_samples = [x for x in pool if x['label'] == 0]
        
        print(f"   📊 总样本池: {len(pool)} (Pos: {len(pos_samples)}, Neg: {len(neg_samples)})")
        
        # 3. 检查数量是否足够构建 Small Set
        if len(pos_samples) < TRAIN_PER_CLASS or len(neg_samples) < TRAIN_PER_CLASS:
            print(f"   ⚠️ 样本不足 (需要各 {TRAIN_PER_CLASS})，跳过！")
            continue
            
        # 4. 采样
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        
        # 抽取训练集
        train_data = pos_samples[:TRAIN_PER_CLASS] + neg_samples[:TRAIN_PER_CLASS]
        random.shuffle(train_data)
        
        # 抽取测试集 (剩下的全部，或者限制数量)
        test_pos = pos_samples[TRAIN_PER_CLASS:]
        test_neg = neg_samples[TRAIN_PER_CLASS:]
        
        # 限制测试集最大数量，防止评估太慢 (例如各取500)
        # 论文好像用了 250+250
        if len(test_pos) > 250: test_pos = test_pos[:250]
        if len(test_neg) > 250: test_neg = test_neg[:250]
        
        test_data = test_pos + test_neg
        random.shuffle(test_data)
        
        # 5. 保存 (注意：文件夹名必须是 task_0, task_1...)
        # 我们按处理顺序赋予 task_id
        final_task_id = processed_count
        save_path_train = os.path.join(OUTPUT_ROOT, f"task_{final_task_id}", "train.json")
        save_path_test = os.path.join(OUTPUT_ROOT, f"task_{final_task_id}", "test.json")
        
        save_json(train_data, save_path_train)
        save_json(test_data, save_path_test)
        
        print(f"   ✅ 生成 Task {final_task_id}: Train({len(train_data)}), Test({len(test_data)})")
        processed_count += 1
        
        # 只需要 10 个任务
        if processed_count >= 10:
            print("\n🛑 已生成 10 个任务，停止处理。")
            break

    # 清理
    shutil.rmtree(TEMP_DIR)
    print(f"\n🎉 数据准备完成！路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()