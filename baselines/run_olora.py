"""
O-LoRA (Orthogonal Subspace Learning) Baseline
================================================
来源：Wang et al., "Orthogonal Subspace Learning for Language Model Continual Learning"
      EMNLP 2023 Findings | arXiv:2310.14152 | GitHub: cmnfriend/O-LoRA

核心机制：
  - 每个新任务训练一套 LoRA 参数（A, B）
  - 对旧任务的 LoRA_A 参数施加 **正交正则化**（Orthogonal Regularization）：
      orth_loss = Σ || A_new · A_old^T ||_1
    强迫新任务的更新方向与历史任务方向正交，减少参数空间干扰
  - 对当前 LoRA 参数施加 L2 正则化，防止偏移过大
  - 总 Loss = CE + λ1 * orth_loss + λ2 * l2_loss

适配说明（相比原论文 T5 版本）：
  - 骨干网络：BERT-base + HuggingFace PEFT LoRA
  - 任务类型：Class-Incremental Learning（CIL），测试时 150 类全竞争
  - 分类头：CosineLinear（与 Sleep-HOP 保持一致，确保公平对比）

使用方式：
  python baselines/run_olora.py --exp_name olora_seed42 --olora_lambda1 1.0 --olora_lambda2 0.1 --epochs 10 --lr 2e-4
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


class OLoRATrainer:
    """
    O-LoRA 训练器：通过正交子空间约束实现无回放的持续学习。

    设计：
    1. old_lora_A_list: 存储所有历史任务的 LoRA_A 参数快照
    2. 每个任务结束后，将当前 LoRA_A 快照添加到历史列表，然后重新初始化 LoRA
    3. 训练时计算新 LoRA_A 与所有旧 LoRA_A 的正交惩罚
    """

    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        # O-LoRA 核心：历史任务 LoRA_A 参数存储
        # 格式: List[ Dict{ param_name -> tensor [r, d_in] } ]
        self.old_lora_A_list = []

        self.lambda1 = args.olora_lambda1  # 正交正则化强度
        self.lambda2 = args.olora_lambda2  # L2 正则化强度

    # -------------------------------------------------------------------------
    # 正交损失：新 LoRA_A 与所有历史 LoRA_A 的点积惩罚
    # -------------------------------------------------------------------------
    def compute_orthogonal_loss(self):
        """
        核心公式：
          orth_loss = Σ_{t<T} Σ_{layer} || A_new · A_old_t^T ||_1

        A_new ∈ R^{r × d_in}，A_old ∈ R^{r × d_in}
        乘积 A_new · A_old^T ∈ R^{r × r}，取绝对值之和
        """
        if not self.old_lora_A_list:
            return torch.tensor(0.0, device=self.device)

        orth_loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if 'lora_A' not in name or not param.requires_grad:
                continue

            for old_snapshot in self.old_lora_A_list:
                if name in old_snapshot:
                    old_A = old_snapshot[name].to(self.device)
                    gram = torch.mm(param, old_A.T)   # [r_new, r_old]
                    orth_loss = orth_loss + torch.abs(gram).sum()

        return orth_loss

    def compute_l2_loss(self):
        """L2 正则化：对所有当前可训练 LoRA 参数施加惩罚"""
        l2_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if ('lora_A' in name or 'lora_B' in name) and param.requires_grad:
                l2_loss = l2_loss + torch.norm(param, p=2)
        return l2_loss

    # -------------------------------------------------------------------------
    # 任务管理：快照 & 重初始化
    # -------------------------------------------------------------------------
    def snapshot_and_reinit_lora(self):
        """
        任务结束时：
        1. 将当前 LoRA_A 参数保存快照（用于后续正交约束）
        2. 重新初始化 LoRA_A/B（为下一任务提供干净起点）
        """
        # Step 1: 保存当前 LoRA_A 快照
        snapshot = {}
        for name, param in self.model.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                snapshot[name] = param.data.clone().cpu()
        self.old_lora_A_list.append(snapshot)

        if not self.args.no_reinit:
            # Step 2: 重新初始化 LoRA 参数
            for name, param in self.model.named_parameters():
                if 'lora_A' in name and param.requires_grad:
                    nn.init.kaiming_uniform_(param.data, a=np.sqrt(5))
                elif 'lora_B' in name and param.requires_grad:
                    nn.init.zeros_(param.data)

        print(f"   📸 [O-LoRA] 已保存 LoRA_A 快照（历史共 {len(self.old_lora_A_list)} 个任务）")
        if not self.args.no_reinit:
            print("   🔄 [O-LoRA] LoRA 参数已重新初始化")

    # -------------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------------
    def train_epoch(self, loader, optimizer, epoch_idx=0):
        self.model.train()
        total_loss, total_ce, total_orth = 0.0, 0.0, 0.0
        has_old = bool(self.old_lora_A_list)

        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            ce_loss = self.criterion(logits, labels)

            orth_loss = self.compute_orthogonal_loss() if has_old else torch.tensor(0.0)
            l2_loss = self.compute_l2_loss()

            loss = ce_loss + self.lambda1 * orth_loss + self.lambda2 * l2_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_orth += orth_loss.item() if has_old else 0.0

            pbar.set_postfix({
                'CE': f"{ce_loss.item():.3f}",
                'Orth': f"{orth_loss.item():.3f}" if has_old else 'N/A'
            })

        n = max(len(loader), 1)
        print(f"    ▶ CE={total_ce/n:.4f} | Orth={total_orth/n:.4f} | Total={total_loss/n:.4f}")

    def train_task(self, loader, task_id):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=0.0
        )

        print(f"🚀 [O-LoRA] Task {task_id} 训练开始，共 {self.args.epochs} 轮 "
              f"(λ1={self.lambda1}, λ2={self.lambda2})...")
        for epoch in range(self.args.epochs):
            self.train_epoch(loader, optimizer, epoch_idx=epoch)

        self.snapshot_and_reinit_lora()

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
    parser = argparse.ArgumentParser(description='O-LoRA Baseline for Continual Learning')
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
    # O-LoRA 专属参数
    parser.add_argument("--olora_lambda1", type=float, default=1.0,
                        help="正交正则化强度 λ1")
    parser.add_argument("--olora_lambda2", type=float, default=0.1,
                        help="L2 正则化强度 λ2")
    parser.add_argument("--no_reinit", action="store_true",
                        help="禁用 LoRA 重初始化（还原原论文行为）")
    parser.add_argument("--no_cosine", action="store_true",
                        help="禁用 CosineLinear")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"🔒 随机种子已锁定: {args.seed}")
    print(f"🚀 O-LoRA Baseline 启动: {args.exp_name}")
    print(f"   λ1={args.olora_lambda1}, λ2={args.olora_lambda2}, "
          f"re-init={'OFF' if args.no_reinit else 'ON'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    trainer = OLoRATrainer(model, device, args)

    R = np.zeros((args.num_tasks, args.num_tasks))
    output_dir = os.path.join("outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 主训练循环
    # =========================================================================
    for task_id in range(args.num_tasks):
        print(f"\n{'='*20} Task {task_id} {'='*20}")

        train_path = os.path.join(args.data_root, f"task_{task_id}", "train.json")
        if not os.path.exists(train_path):
            print(f"⚠️  训练数据不存在: {train_path}")
            continue

        train_loader = DataLoader(
            JSONLDataset(train_path, tokenizer),
            batch_size=args.batch_size, shuffle=True
        )

        trainer.train_task(train_loader, task_id=task_id)

        # 评估所有已学任务
        print(f"🧐 评估 Task 0 -> {task_id}...")
        test_accs = []
        with torch.no_grad():
            for eval_id in range(task_id + 1):
                test_path = os.path.join(args.data_root, f"task_{eval_id}", "test.json")
                if not os.path.exists(test_path):
                    test_accs.append(0.0)
                    continue
                test_loader = DataLoader(
                    JSONLDataset(test_path, tokenizer), batch_size=64
                )
                acc = trainer.evaluate(test_loader)
                test_accs.append(acc)
                R[task_id, eval_id] = acc

        avg_acc = np.mean(test_accs) * 100
        print(f"📊 Task {task_id} 成绩单:")
        print(f"   > Average Accuracy: {avg_acc:.2f}%")
        print(f"   > Acc List: {[f'{x*100:.1f}' for x in test_accs]}")

    # =========================================================================
    # 最终结算
    # =========================================================================
    if args.num_tasks > 0:
        final_idx = args.num_tasks - 1
        final_avg = np.mean(R[final_idx, :args.num_tasks])
        bwt = (
            np.mean([R[final_idx, i] - R[i, i] for i in range(args.num_tasks - 1)])
            if args.num_tasks > 1 else 0.0
        )

        results = {
            "method": "O-LoRA",
            "exp_name": args.exp_name,
            "seed": args.seed,
            "olora_lambda1": args.olora_lambda1,
            "olora_lambda2": args.olora_lambda2,
            "no_reinit": args.no_reinit,
            "final_avg": float(final_avg),
            "bwt": float(bwt),
            "matrix": R.tolist()
        }
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print("\n" + "🏆" * 10)
        print(f"[O-LoRA] 最终平均准确率 (Final Avg): {final_avg*100:.2f}%")
        print(f"[O-LoRA] 向后遗忘率 (BWT):           {bwt*100:.2f}%")
        print(f"[O-LoRA] 结果已保存至: {results_path}")
        print("🏆" * 10)


if __name__ == "__main__":
    main()
