import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from src.sleep import sleep_phase

class HOPTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        self.task_importance = {}       # 字典 A：当前任务活跃度积累器（每次新任务前清零）
        self.lifetime_elite_mask = {}   # 字典 B：终身精英保护图（绝对 0/1，只增不减）
        self.trainable_params_cache = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.task_importance[name] = torch.zeros_like(param.data)
                self.lifetime_elite_mask[name] = torch.zeros_like(param.data)
                self.trainable_params_cache.append((name, param))

    def train_epoch(self, loader, optimizer, epoch_idx=0):
        self.model.train()
        total_loss = 0

        # 预计算：是否存在历史精英（Task 0 时全为 0，跳过所有保护逻辑）
        has_elite = bool(self.lifetime_elite_mask) and any(
            m.any().item() for m in self.lifetime_elite_mask.values()
        )

        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)

        for batch in pbar:
            input_ids      = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels         = batch['labels'].to(self.device)

            optimizer.zero_grad()

            # ☀️ [Wake] 纯净新任务拟合 —— 唯一 Loss，无任何历史回放干扰
            logits = self.model(input_ids, attention_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()

            total_loss += loss.item()

            # [Hebbian Accumulation] 累加梯度绝对值作为 Taylor 重要性估计
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None:
                        self.task_importance[name] += param.grad.abs()

            # 🔒 [Hard Subnetwork Masking] 绝对物理梯度隔离
            # 精英(mask=1)：梯度强制清零，任何新任务学习绝对无法侵入
            # 非精英(mask=0)：梯度完整保留，新任务在此自由生长
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None and name in self.lifetime_elite_mask:
                        param.grad *= (1.0 - self.lifetime_elite_mask[name].to(param.device))

            # ✅ [Adam 泄漏修复] Pre-Step 快照 → Post-Step 精英位置强制还原
            # 根因：即使梯度清零，Adam 内已积累的历史动量 m_t 仍会施加残余位移
            # 修复：Step 前克隆精英权重，Step 后将精英位置强制还原为快照值
            if has_elite:
                elite_snapshot = {
                    name: param.data.clone()
                    for name, param in self.trainable_params_cache
                    if name in self.lifetime_elite_mask
                }

            optimizer.step()

            if has_elite:
                with torch.no_grad():
                    for name, param in self.trainable_params_cache:
                        if name in elite_snapshot:
                            mask = self.lifetime_elite_mask[name].to(param.device)
                            param.data.copy_(
                                mask * elite_snapshot[name] + (1.0 - mask) * param.data
                            )

            pbar.set_postfix({'Loss': f"{loss.item():.3f}"})

    def train_task(self, loader):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=0.0
        )

        # ✅ [双字典] 只清零当前任务积累器，终身精英掩码绝不触动
        for k in self.task_importance:
            self.task_importance[k].zero_()

        print(f"🚀 [Wake] 开始清醒学习阶段，共 {self.args.epochs} 轮...")

        for epoch in range(self.args.epochs):
            self.train_epoch(loader, optimizer, epoch_idx=epoch)

        # ✅ [双字典] 从本任务 task_importance 独立提取精英，永久固化入终身二值掩码
        # 精英选取在本任务内部完成（Top elite_quantile），T1 的梯度量级永远无法影响 T0 精英资格
        # max 合并：一旦当过精英（=1），永不降级
        elite_q      = getattr(self.args, 'elite_quantile', 0.85)
        lora_q       = getattr(self.args, 'lora_elite_quantile', None) or elite_q
        elite_pct    = (1.0 - elite_q) * 100
        lora_pct     = (1.0 - lora_q) * 100
        print(f"📚 [Lifetime Elite] 固化精英: 通用 Top{elite_pct:.0f}% / LoRA Top{lora_pct:.0f}% (quantile 通用={elite_q}, LoRA={lora_q})...")
        with torch.no_grad():
            for k in self.task_importance:
                imp = self.task_importance[k]

                # 根据参数类型选择对应分位数
                q = lora_q if 'lora' in k else elite_q

                if imp.max() > 0:
                    # 分类器：按行提取精英（行 = 类别节点）
                    if 'classifier' in k and imp.dim() > 1:
                        row_imp = imp.mean(dim=1, keepdim=True)
                        threshold = torch.quantile(row_imp.float(), q)
                        current_elite = (row_imp > threshold).float().expand_as(imp)
                    else:
                        threshold = torch.quantile(imp.float(), q)
                        current_elite = (imp > threshold).float()
                else:
                    current_elite = torch.zeros_like(imp)

                if k in self.lifetime_elite_mask:
                    self.lifetime_elite_mask[k] = torch.max(
                        self.lifetime_elite_mask[k],
                        current_elite
                    )

        torch.cuda.empty_cache()
        gc.collect()

    def _merge_and_reinit_lora(self):
        """
        🧬 LoRA Merge & Re-init (短期记忆 → 长期记忆固化)
        
        仿生学原理：白天学到的短期突触变化 (LoRA) 在睡眠时
        被固化为长期记忆 (BERT 主干权重)，然后突触重置为基线，
        为明天学习新知识做准备。
        
        数学本质：
          W_base_new = W_base + B @ A * scaling   (合并)
          A, B = 0                                  (重置)
          下一任务前向: y = x @ W_base_new + x @ 0 = x @ W_base_new
        """
        merged_count = 0
        with torch.no_grad():
            for module in self.model.bert.modules():
                # 检测 PEFT LoRA 层
                if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
                    continue
                if not hasattr(module, 'scaling'):
                    continue

                for adapter_name in list(module.lora_A.keys()):
                    lora_A_weight = module.lora_A[adapter_name].weight  # [rank, in]
                    lora_B_weight = module.lora_B[adapter_name].weight  # [out, rank]
                    scaling = module.scaling[adapter_name]

                    # 获取 base layer 权重（冻结的预训练参数）
                    if hasattr(module, 'get_base_layer'):
                        base_weight = module.get_base_layer().weight
                    elif hasattr(module, 'base_layer'):
                        base_weight = module.base_layer.weight
                    else:
                        continue

                    # 合并：W_base += B @ A * scaling
                    base_weight.data += (lora_B_weight @ lora_A_weight) * scaling

                    # 重置 LoRA 为零（突触归零，准备学新知识）
                    lora_A_weight.zero_()
                    lora_B_weight.zero_()
                    merged_count += 1

        # 清除 LoRA 的精英掩码（旧 LoRA 已合并入主干，掩码失去意义）
        # 分类器精英掩码保留不动
        for name in self.lifetime_elite_mask:
            if 'lora' in name:
                self.lifetime_elite_mask[name].zero_()

        print(f"   🧬 [Consolidation] {merged_count} 个 LoRA 层已合并入主干并重置为零")

    def sleep(self, tokenizer, current_mask, prototype_memory, prototype_loader=None):
        if not self.args.use_sleep:
            return self.model, current_mask

        print(f"💤 [Sleep] 进入睡眠巩固阶段...")

        # 🧬 Phase 0: LoRA → Base 合并（短期记忆固化为长期记忆）
        self._merge_and_reinit_lora()

        config = {
            'alpha':    self.args.alpha,
            'target_norm': self.args.target_norm,
            'no_rem':   getattr(self.args, 'no_rem', False),
            'lora_alpha': getattr(self.args, 'lora_alpha', 0.01),
        }

        self.model, new_mask = sleep_phase(
            self.model,
            tokenizer,
            self.device,
            config,
            self.lifetime_elite_mask,
            prototype_memory,
            previous_mask=current_mask,
            prototype_loader=prototype_loader,
        )

        if new_mask is not None:
            if current_mask is None:
                current_mask = new_mask
            else:
                for k, v in new_mask.items():
                    current_mask[k] = v

            total_params  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if total_params > 0:
                locked_params = sum(m.sum().item() for m in current_mask.values())
                print(f"🛡️  [Memory Capacity] 已保护神经元: {locked_params/total_params*100:.2f}%")

        return self.model, current_mask

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