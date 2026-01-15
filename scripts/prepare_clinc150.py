import os
import json
import random
import urllib.request

# === 配置 ===
DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
OUTPUT_ROOT = "data/clinc150"
TASKS_NUM = 15
CLASSES_PER_TASK = 10
SEED = 42

def download_data():
    if not os.path.exists("temp_clinc.json"):
        print("📥 正在下载 CLINC150...")
        urllib.request.urlretrieve(DATA_URL, "temp_clinc.json")
    with open("temp_clinc.json", "r") as f:
        return json.load(f)

def save_json(data_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")

def main():
    print(f"🚀 准备 CLINC150 数据 (15 Tasks, 150 Classes)")
    raw_data = download_data()

    all_train = raw_data['train']
    all_test = raw_data['test']

    labels = sorted(list(set([item[1] for item in all_train])))
    label2id = {label: i for i, label in enumerate(labels)}

    train_by_label = {label: [] for label in labels}
    test_by_label = {label: [] for label in labels}

    for text, label in all_train:
        train_by_label[label].append({'text': text, 'label': label2id[label]})
    for text, label in all_test:
        test_by_label[label].append({'text': text, 'label': label2id[label]})

    random.seed(SEED)
    random.shuffle(labels)

    for task_id in range(TASKS_NUM):
        start_idx = task_id * CLASSES_PER_TASK
        end_idx = start_idx + CLASSES_PER_TASK
        task_labels = labels[start_idx:end_idx]

        if not task_labels: break

        task_train = []
        task_test = []

        for label in task_labels:
            task_train.extend(train_by_label[label])
            task_test.extend(test_by_label[label])

        random.shuffle(task_train)
        random.shuffle(task_test)

        save_json(task_train, os.path.join(OUTPUT_ROOT, f"task_{task_id}", "train.json"))
        save_json(task_test, os.path.join(OUTPUT_ROOT, f"task_{task_id}", "test.json"))
        print(f"   ✅ Task {task_id} 生成完毕")

    print(f"\n🎉 CLINC150 数据准备完成！")

if __name__ == "__main__":
    main()