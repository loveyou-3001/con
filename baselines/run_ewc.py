"""
EWC (Elastic Weight Consolidation) Baseline
============================================
经典持续学习正则化方法，用于与 Sleep-HOP 做对比实验。

原理：
  - 每个任务结束后，计算 Fisher 信息矩阵（对角近似）
  - 下一任务训练时，对重要参数施加二次正则惩罚
  - Loss = CE(y, ŷ) + λ/2 * Σ F_i * (θ_i - θ*_i)²

使用方式：
  python baselines/run_ewc.py --exp_name ewc_baseline --ewc_lambda 5000 --epochs 10 --lr 2e-4
"""

import os, sys, json, torch, gc, argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import HOPBertClassifier
from src.dataset import JSONLDataset


class EWCTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        # EWC 核心存储
        self.fisher = {}           # 累积 Fisher 信息矩阵（对角近似）
        self.optimal_params = {}   # 上一任务结束后的最优参数快照
        self.ewc_lambda = args.ewc_lambda

    def compute_fisher(self, loader):
        """计算当前任务的 Fisher 信息矩阵（对角近似）"""
        self.model.eval()
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        num_samples = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.model.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

            num_samples += len(labels)

        for n in fisher:
            fisher[n] /= max(num_samples, 1)

        return fisher

    def update_ewc_data(self, loader):
        """任务结束后更新 Fisher 和最优参数快照"""
        print("   📐 [EWC] 计算 Fisher 信息矩阵...")
        new_fisher = self.compute_fisher(loader)

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.fisher:
                # Online EWC: 累积 Fisher
                self.fisher[n] = self.fisher[n] + new_fisher.get(n, torch.zeros_like(p.data))
            else:
                self.fisher[n] = new_fisher.get(n, torch.zeros_like(p.data))

            self.optimal_params[n] = p.data.clone()

    def ewc_loss(self):
        """计算 EWC 正则化损失"""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.fisher and n in self.optimal_params:
                loss += (self.fisher[n].to(p.device) * (p - self.optimal_params[n].to(p.device)) ** 2).sum()
        return self.ewc_lambda / 2 * loss

    def train_epoch(self, loader, optimizer, epoch_idx=0):
        self.model.train()
        total_loss = 0
        has_ewc = bool(self.fisher)

        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            ce_loss = self.criterion(logits, labels)

            if has_ewc:
                ewc_reg = self.ewc_loss()
                loss = ce_loss + ewc_reg
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.3f}"})

    def train_task(self, loader):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=0.0
        )

        print(f"🚀 [EWC] 开始训练，共 {self.args.epochs} 轮 (λ={self.ewc_lambda})...")
        for epoch in range(self.args.epochs):
            self.train_epoch(loader, optimizer, epoch_idx=epoch)

        # 任务结束后更新 Fisher 和参数快照
        self.update_ewc_data(loader)

        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self, loader):
        self.model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in loader:
                logits = self.model(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                gts.extend(batch['labels'].cpu().numpy())

        from sklearn.metrics import accuracy_score
        return accuracy_score(gts, preds)


def parse_args():
    parser = argparse.ArgumentParser(description='EWC Baseline for Continual Learning')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--model_id", type=str, default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_tasks", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--ewc_lambda", type=float, default=5000, help="EWC 正则化强度")
    parser.add_argument("--no_cosine", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # 锁定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"🔒 随机种子已锁定: {args.seed}")
    print(f"🚀 EWC Baseline 启动: {args.exp_name} (λ={args.ewc_lambda})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型路径
    from src.model import get_bert_path
    model_path = get_bert_path(args.model_id)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = HOPBertClassifier(
        model_path,
        num_classes=args.num_classes,
        use_lora=True,
        use_cosine=not args.no_cosine,
        lora_rank=args.lora_rank
    ).to(device)

    trainer = EWCTrainer(model, device, args)

    R = np.zeros((args.num_tasks, args.num_tasks))
    output_dir = os.path.join("outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # 主循环
    for task_id in range(args.num_tasks):
        print(f"\n{'='*20} Task {task_id} {'='*20}")

        train_path = os.path.join(args.data_root, f"task_{task_id}", "train.json")
        if not os.path.exists(train_path):
            print(f"⚠️  训练数据不存在: {train_path}")
            continue
        train_loader = DataLoader(JSONLDataset(train_path, tokenizer),
                                  batch_size=args.batch_size, shuffle=True)

        # 训练（含 EWC 正则化）
        trainer.train_task(train_loader)

        # 评估所有已学任务
        print(f"🧐 正在评估历史任务性能 (Task 0 -> {task_id})...")
        test_accs = []
        model.eval()
        with torch.no_grad():
            for eval_id in range(task_id + 1):
                test_path = os.path.join(args.data_root, f"task_{eval_id}", "test.json")
                if not os.path.exists(test_path):
                    test_accs.append(0.0)
                    continue
                test_loader = DataLoader(JSONLDataset(test_path, tokenizer), batch_size=64)
                acc = trainer.evaluate(test_loader)
                test_accs.append(acc)
                R[task_id, eval_id] = acc

        avg_acc = np.mean(test_accs) * 100
        print(f"📊 Task {task_id} 完结成绩单:")
        print(f"   > Average Accuracy: {avg_acc:.2f}%")
        print(f"   > Acc List: {[f'{x*100:.1f}' for x in test_accs]}")

    # 最终结算
    if args.num_tasks > 0:
        final_idx = args.num_tasks - 1
        final_avg = np.mean(R[final_idx, :args.num_tasks])
        bwt = np.mean([R[final_idx, i] - R[i, i] for i in range(args.num_tasks - 1)]) if args.num_tasks > 1 else 0

        results = {"final_avg": float(final_avg), "bwt": float(bwt), "matrix": R.tolist()}
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        print("\n" + "🏆" * 10)
        print(f"[EWC] 最终平均准确率 (Final Avg): {final_avg*100:.2f}%")
        print(f"[EWC] 向后遗忘率 (BWT): {bwt*100:.2f}%")
        print("🏆" * 10)


if __name__ == "__main__":
    main()
