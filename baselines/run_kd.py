"""
KD/LwF (Knowledge Distillation / Learning without Forgetting) Baseline
=======================================================================
来源：Li & Hoiem, "Learning without Forgetting", TPAMI 2017

核心机制：
  - 每个新任务训练前，冻结旧模型作为 Teacher
  - 训练时在旧类 logits 上用 KL 散度蒸馏旧知识
  - Loss = CE(y, ŷ) + α * KD_loss(softmax(z_old/T), softmax(z_teacher/T))

使用方式：
  python baselines/run_kd.py --exp_name kd_seed42 --kd_alpha 1.0 --kd_temp 2.0 --epochs 10 --lr 2e-4
"""

import os, sys, json, torch, gc, argparse, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import HOPBertClassifier
from src.dataset import JSONLDataset


class KDTrainer:
    """
    KD/LwF 训练器：通过知识蒸馏保留旧任务知识。

    设计：
    1. 每个新任务开始前，将当前模型深拷贝为 teacher（冻结）
    2. 训练时计算 CE + KD loss
    3. KD loss = KL(softmax(student_logits/T), softmax(teacher_logits/T))
       仅在 teacher 已见过的类别上计算
    """

    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        self.teacher = None
        self.num_old_classes = 0  # Teacher 模型已见过的类别数上限
        self.kd_alpha = args.kd_alpha
        self.kd_temp = args.kd_temp

    def snapshot_teacher(self, num_classes_seen):
        """任务开始前，冻结当前模型作为 Teacher"""
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.num_old_classes = num_classes_seen
        print(f"   📸 [KD] Teacher 模型已冻结（覆盖 {self.num_old_classes} 个旧类）")

    def compute_kd_loss(self, student_logits, input_ids, attention_mask):
        """
        计算知识蒸馏损失：
        在旧类的 logits 上，用 KL 散度迫使 Student 保持 Teacher 的输出分布
        """
        if self.teacher is None or self.num_old_classes == 0:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_logits = self.teacher(input_ids, attention_mask)

        # 只在旧类上蒸馏（前 num_old_classes 个 logit）
        old_student = student_logits[:, :self.num_old_classes]
        old_teacher = teacher_logits[:, :self.num_old_classes]

        T = self.kd_temp
        kd_loss = F.kl_div(
            F.log_softmax(old_student / T, dim=1),
            F.softmax(old_teacher / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        return kd_loss

    def train_epoch(self, loader, optimizer, epoch_idx=0):
        self.model.train()
        total_loss, total_ce, total_kd = 0.0, 0.0, 0.0
        has_teacher = self.teacher is not None and self.num_old_classes > 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            ce_loss = self.criterion(logits, labels)

            if has_teacher:
                kd_loss = self.compute_kd_loss(logits, input_ids, attention_mask)
                loss = ce_loss + self.kd_alpha * kd_loss
                total_kd += kd_loss.item()
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()

            pbar.set_postfix({
                'CE': f"{ce_loss.item():.3f}",
                'KD': f"{kd_loss.item():.3f}" if has_teacher else 'N/A'
            })

        n = max(len(loader), 1)
        print(f"    ▶ CE={total_ce/n:.4f} | KD={total_kd/n:.4f} | Total={total_loss/n:.4f}")

    def train_task(self, loader, task_id, classes_per_task=10):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=0.0
        )

        print(f"🚀 [KD] Task {task_id} 训练开始，共 {self.args.epochs} 轮 "
              f"(α={self.kd_alpha}, T={self.kd_temp})...")
        for epoch in range(self.args.epochs):
            self.train_epoch(loader, optimizer, epoch_idx=epoch)

        # 任务结束后：更新 Teacher 为当前模型
        num_classes_seen = (task_id + 1) * classes_per_task
        self.snapshot_teacher(num_classes_seen)

        # 释放旧 Teacher 的显存
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
    parser = argparse.ArgumentParser(description='KD/LwF Baseline for Continual Learning')
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
    parser.add_argument("--classes_per_task", type=int, default=10,
                        help="每任务类别数（CLINC-150: 150/15=10）")
    # KD 专属参数
    parser.add_argument("--kd_alpha", type=float, default=1.0,
                        help="蒸馏损失权重 α")
    parser.add_argument("--kd_temp", type=float, default=2.0,
                        help="蒸馏温度 T")
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
    print(f"🚀 KD/LwF Baseline 启动: {args.exp_name}")
    print(f"   α={args.kd_alpha}, T={args.kd_temp}")

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

    trainer = KDTrainer(model, device, args)

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

        trainer.train_task(train_loader, task_id=task_id,
                           classes_per_task=args.classes_per_task)

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
            "method": "KD/LwF",
            "exp_name": args.exp_name,
            "seed": args.seed,
            "kd_alpha": args.kd_alpha,
            "kd_temp": args.kd_temp,
            "final_avg": float(final_avg),
            "bwt": float(bwt),
            "matrix": R.tolist()
        }
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print("\n" + "🏆" * 10)
        print(f"[KD/LwF] 最终平均准确率 (Final Avg): {final_avg*100:.2f}%")
        print(f"[KD/LwF] 向后遗忘率 (BWT):           {bwt*100:.2f}%")
        print(f"[KD/LwF] 结果已保存至: {results_path}")
        print("🏆" * 10)


if __name__ == "__main__":
    main()
