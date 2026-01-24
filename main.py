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
    parser = argparse.ArgumentParser(description="Sleep-HOP: PCGrad & Asymmetric Routing")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    
    # [新增] 显式支持 model_id 防止报错
    parser.add_argument("--model_id", type=str, default="bert-base-uncased")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--hop_order", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_sleep", action="store_true")
    
    # Sleep & Mechanism 参数
    parser.add_argument("--threshold", type=float, default=50) 
    parser.add_argument("--target_norm", type=float, default=11.5)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--feat_lambda", type=float, default=5.0)
    parser.add_argument("--proto_lambda", type=float, default=2.0) # 建议设为 2.0 增强复习
    parser.add_argument("--kd_lambda", type=float, default=0.0)    
    
    return parser.parse_args()

def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = os.path.join("output", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        
    print(f"🚀 实验启动: {args.exp_name} (Modular Optimized)")
    
    # 初始化模型
    # 使用 args.model_id 更加灵活
    model_path = get_bert_path(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = HOPBertClassifier(model_path, num_classes=args.num_classes, hop_order=args.hop_order, use_lora=True).to(device)
    
    prototype_memory = PrototypeMemory(args.num_classes, 768, device)
    trainer = HOPTrainer(model, device, args) 
    
    R = np.zeros((args.num_tasks, args.num_tasks))
    current_mask = None

    # === 持续学习循环 ===
    for task_id in range(args.num_tasks):
        print(f"\n=== Task {task_id} ===")
        
        # 1. 准备数据
        train_path = os.path.join(args.data_root, f"task_{task_id}", "train.json")
        if not os.path.exists(train_path): 
            print(f"⚠️ 数据文件不存在: {train_path}，停止实验。")
            break
            
        train_dataset = JSONLDataset(train_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # 2. 训练 (委托给 Trainer)
        # Trainer 内部会自动处理 Stream A (新任务) 和 Stream B (复习)
        trainer.train_task(train_loader, prototype_memory, current_mask)
        
        # 3. 更新记忆 (海马体)
        print("🧠 [Hippocampus] 正在更新原型记忆...")
        prototype_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        prototype_memory.update_prototypes(model, prototype_loader, device)

        # 4. 睡眠阶段 (Sleep Phase)
        # [CRITICAL FIX] 必须传入 prototype_memory，否则 REM 梦境无法运行
        model, current_mask = trainer.sleep(tokenizer, current_mask, prototype_memory)
        
        # 5. 保存检查点
        task_ckpt_dir = os.path.join(output_dir, f"task_{task_id}")
        model.bert.save_pretrained(task_ckpt_dir)
        torch.save(model.classifier.state_dict(), os.path.join(task_ckpt_dir, "head.pth"))
        
        # 6. 评估所有历史任务
        print(f"🧐 正在评估 Task 0 -> Task {task_id}...")
        accs = []
        for eval_id in range(task_id + 1):
            test_path = os.path.join(args.data_root, f"task_{eval_id}", "test.json")
            test_loader = DataLoader(JSONLDataset(test_path, tokenizer), batch_size=64)
            
            # 临时加载模型进行评估
            # 注意：评估时不需要 LoRA 的梯度，也不需要 CosineLinear 的 Sigma 梯度
            # 动态计算类别数防止越界
            labels_in_task = set()
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            labels_in_task.add(item['label'])
                task_num_classes = max(labels_in_task) + 1
            except:
                task_num_classes = args.num_classes

            # 实例化评估模型
            eval_model = HOPBertClassifier(model_path, num_classes=args.num_classes, hop_order=args.hop_order, use_lora=False)
            
            # 加载 LoRA 权重
            eval_ckpt_dir = os.path.join(output_dir, f"task_{eval_id}")
            eval_model.bert = PeftModel.from_pretrained(eval_model.bert, eval_ckpt_dir, is_trainable=False)
            # 加载 Classifier 权重
            eval_model.classifier.load_state_dict(torch.load(os.path.join(eval_ckpt_dir, "head.pth")))
            
            eval_model.to(device)
            eval_model.eval()
            
            # 使用 Trainer 的 evaluate 方法
            eval_trainer = HOPTrainer(eval_model, device, args)
            acc = eval_trainer.evaluate(test_loader)
            
            del eval_model, eval_trainer
            torch.cuda.empty_cache()
            
            accs.append(acc)
            R[task_id, eval_id] = acc
        
        print(f"👉 Avg Acc: {np.mean(accs)*100:.2f}% | Task 0 Acc: {accs[0]*100:.2f}%")
        print(f"📊 { [f'{x*100:.1f}' for x in accs] }")

    # === 实验结束结算 ===
    if args.num_tasks > 0:
        final_avg = np.mean(R[args.num_tasks-1, :args.num_tasks])
        bwt = np.mean([R[args.num_tasks-1, i] - R[i, i] for i in range(args.num_tasks - 1)])
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({"final_avg": final_avg, "bwt": bwt, "R": R.tolist()}, f, indent=4)
        print(f"\n🏆 Final Avg: {final_avg*100:.2f}% | BWT: {bwt*100:.2f}%")

if __name__ == "__main__":
    main()