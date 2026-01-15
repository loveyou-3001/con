import torch
import os
import argparse
import sys
import matplotlib.pyplot as plt

# 添加 src 目录到路径，确保能导入 model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import HOPBertClassifier, get_bert_path
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="检查实验结果的权重范数变化")
    parser.add_argument("--exp_name", type=str, required=True, help="实验文件夹名称，例如 20news_baseline_hop1")
    parser.add_argument("--hop_order", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=10, help="总任务数")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = os.path.join("output", args.exp_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🕵️ 正在分析实验: {args.exp_name}")
    print(f"📂 路径: {output_dir}\n")

    # 初始化基座
    model_path = get_bert_path()
    # 只需要构建一次结构，后面只加载权重
    base_model = HOPBertClassifier(model_path, num_classes=2, hop_order=args.hop_order, use_lora=False)
    
    norms = []
    steps = []

    print("| Task ID | Weight Norm (L2) | 状态 |")
    print("|---------|------------------|------|")

    for task_id in range(args.tasks):
        task_dir = os.path.join(output_dir, f"task_{task_id}")
        head_path = os.path.join(task_dir, "head.pth")
        
        if not os.path.exists(task_dir):
            print(f"⚠️ 找不到 Task {task_id} 的记录，停止。")
            break

        # 1. 加载 LoRA (这一步对于计算 Head 的 Norm 不是必须的，但为了完整性)
        # model = PeftModel.from_pretrained(base_model.bert, task_dir)
        
        # 2. 加载 Head (我们主要关注分类头的权重膨胀)
        if os.path.exists(head_path):
            state_dict = torch.load(head_path, map_location='cpu')
            # 提取第一层线性层的权重 (连接 HOP 特征的那一层)
            # 键名通常是 '0.weight' (因为 classifier 是 Sequential)
            weight = state_dict.get('0.weight')
            
            if weight is not None:
                norm_val = weight.norm().item()
                norms.append(norm_val)
                steps.append(task_id)
                
                status = "🟢 健康" if norm_val < 11.0 else "🔴 膨胀"
                print(f"| {task_id:7d} | {norm_val:16.4f} | {status} |")
            else:
                print(f"| {task_id:7d} | {'Error':16} | ❌ 结构不符 |")
        else:
            print(f"| {task_id:7d} | {'Missing':16} | ❌ 缺失 Head |")

    # (可选) 简单的画图
    print(f"\n📈 范数趋势: {norms}")
    
    # 保存范数图
    plt.figure(figsize=(8, 5))
    plt.plot(steps, norms, marker='o', linestyle='-', label=args.exp_name)
    plt.title(f"Weight Norm Evolution - {args.exp_name}")
    plt.xlabel("Task ID")
    plt.ylabel("L2 Norm of Classifier Weights")
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(output_dir, "norm_curve.png")
    plt.savefig(save_path)
    print(f"🖼️ 范数变化图已保存至: {save_path}")

if __name__ == "__main__":
    main()