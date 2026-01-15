import os
import subprocess
import shutil

REPO_URL = "https://github.com/ZixuanKe/LifelongSentClass.git"
TEMP_DIR = "debug_dsc_repo"

def main():
    print(f"🕵️‍♀️ 正在进行数据格式诊断...")
    
    # 1. Clone 仓库
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    subprocess.check_call(f"git clone {REPO_URL} {TEMP_DIR}", shell=True)
    
    # 2. 找到一个训练文件
    target_file = os.path.join(TEMP_DIR, "dat/dsc/Amazon_Instant_Video_train.tsv")
    
    if not os.path.exists(target_file):
        # 尝试备用路径 (有时候作者会改目录结构)
        target_file = os.path.join(TEMP_DIR, "data/dsc/Amazon_Instant_Video_train.tsv")

    if not os.path.exists(target_file):
        print("❌ 找不到目标文件，请检查目录结构：")
        print(subprocess.check_output(f"ls -R {TEMP_DIR}", shell=True).decode())
        return

    # 3. 打印前 10 行
    print(f"\n📄 文件内容预览 ({target_file}):")
    print("="*40)
    with open(target_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            # 使用 repr() 打印，这样能看到隐藏字符（如 \t, \n）
            print(f"Line {i}: {repr(line)}")
    print("="*40)
    
    # 4. 清理
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()