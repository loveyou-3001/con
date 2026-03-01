import argparse
import os
import torch
import numpy as np
import json
import random
import copy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel

# 引入模块
from src.model import HOPBertClassifier, get_bert_path
from src.dataset import JSONLDataset
from src.memory import PrototypeMemory
from src.trainer import HOPTrainer 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 随机种子已锁定: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sleep-HOP: Orthogonal Synaptic Homeostasis")
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称，用于保存输出")
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--model_id", type=str, default="bert-base-uncased")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--hop_order", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_sleep", action="store_true", help="是否启用睡眠机制")
    
    # --- Sleep & Mechanism 核心参数 ---
    parser.add_argument("--target_norm", type=float, default=11.5, help="NREM 权重天花板")
    parser.add_argument("--alpha", type=float, default=0.5, help="NREM 压缩强度")
    parser.add_argument("--proto_lambda", type=float, default=2.0, help="Wake 阶段原型回放强度")
    
    # --- 消融实验专用开关 ---
    parser.add_argument("--no_rem", action="store_true", help="[Ablation A3] 跳过 REM 梦境修复")
    parser.add_argument("--lora_alpha", type=float, default=0.01, help="[Ablation A4] LoRA 软豁免强度")
    parser.add_argument("--no_cosine", action="store_true", help="[Ablation A1] 禁用 CosineLinear")
    parser.add_argument('--distill_lambda', type=float, default=2.0, help='潜意识蒸馏(Subconscious Distillation)的Loss权重')    
    
    return parser.parse_args()

def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 环境准备
    output_dir = os.path.join("output", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        
    print(f"🚀 实验启动: {args.exp_name} (Modular Optimized)")
    
    # 2. 初始化组件
    model_path = get_bert_path(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = HOPBertClassifier(
        model_path, 
        num_classes=args.num_classes, 
        hop_order=args.hop_order, 
        use_lora=True,
        use_cosine=not args.no_cosine
    ).to(device)

    prototype_memory = PrototypeMemory(args.num_classes, 768, device)
    trainer = HOPTrainer(model, device, args) 

    R = np.zeros((args.num_tasks, args.num_tasks))
    current_mask = None

    # 3. 持续学习循环
    for task_id in range(args.num_tasks):
        print(f"\n" + "="*20 + f" Task {task_id} " + "="*20)
        
        # --- A. 准备数据 ---
        train_path = os.path.join(args.data_root, f"task_{task_id}", "train.json")
        if not os.path.exists(train_path): 
            print(f"⚠️ 找不到任务数据: {train_path}，跳过该任务。")
            continue
            
        train_dataset = JSONLDataset(train_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # --- B. 清醒学习 (Wake Phase) ---
        print(f"☀️ [Wake] 正在训练新任务...")
        trainer.train_task(train_loader, prototype_memory, current_mask)
        
        # --- C. 记忆巩固 (Update Prototypes) ---
        # 使用当前任务的数据集生成高斯原型
        print("🧠 [Hippocampus] 正在更新高斯原型记忆...")
        prototype_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        prototype_memory.update_prototypes(model, prototype_loader, device)

        # --- D. 睡眠阶段 (Sleep Phase: NREM + REM) ---
        if args.use_sleep:
            # 这里的 trainer.sleep 内部会自动调用我们修复过的 synaptic_downscaling
            model, current_mask = trainer.sleep(tokenizer, current_mask, prototype_memory)
        
        # --- E. 保存进度 (Checkpoints) ---
        task_ckpt_dir = os.path.join(output_dir, f"task_{task_id}")
        os.makedirs(task_ckpt_dir, exist_ok=True)
        # 只保存增量部分：LoRA 和 Classifier Head
        model.bert.save_pretrained(task_ckpt_dir)
        torch.save(model.classifier.state_dict(), os.path.join(task_ckpt_dir, "head.pth"))
        
        # --- F. 评估所有已学任务 (Evaluate All Learned Tasks) ---
        print(f"🧐 正在评估历史任务性能 (Task 0 -> {task_id})...")
        accs = []
        model.eval() # 切换到评估模式
        
        with torch.no_grad():
            for eval_id in range(task_id + 1):
                test_path = os.path.join(args.data_root, f"task_{eval_id}", "test.json")
                if not os.path.exists(test_path):
                    accs.append(0.0)
                    continue
                
                test_loader = DataLoader(JSONLDataset(test_path, tokenizer), batch_size=64)
                # 🌟 优化：直接使用内存中的模型进行评估，避免重复加载导致的显存溢出
                acc = trainer.evaluate(test_loader)
                accs.append(acc)
                R[task_id, eval_id] = acc
        
        # 实时打印成绩单
        avg_acc = np.mean(accs) * 100
        print(f"📊 Task {task_id} 完结成绩单:")
        print(f"   > Average Accuracy: {avg_acc:.2f}%")
        print(f"   > Current Acc List: {[f'{x*100:.1f}' for x in accs]}")

    # 4. 最终结算
    if args.num_tasks > 0:
        final_idx = args.num_tasks - 1
        final_avg = np.mean(R[final_idx, :args.num_tasks])
        # 计算向后遗忘率 (BWT): 越接近 0 越好
        bwt = np.mean([R[final_idx, i] - R[i, i] for i in range(args.num_tasks - 1)]) if args.num_tasks > 1 else 0
        
        results = {
            "final_avg": float(final_avg),
            "bwt": float(bwt),
            "matrix": R.tolist()
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        print("\n" + "🏆" * 10)
        print(f"最终平均准确率 (Final Avg): {final_avg*100:.2f}%")
        print(f"向后遗忘率 (BWT): {bwt*100:.2f}%")
        print("🏆" * 10)

if __name__ == "__main__":
    main()