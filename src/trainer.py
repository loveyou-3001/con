import torch
import torch.nn as nn
import gc
import copy
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

    def train_epoch(self, loader, prototype_memory, previous_mask, optimizer, frozen_model=None, epoch_idx=0):
        self.model.train()
        total_loss = 0
        valid_batches = 0

        proto_lambda = getattr(self.args, 'proto_lambda', 1.0)
        distill_lambda = getattr(self.args, 'distill_lambda', 1.0)

        # 预计算：当前是否存在历史精英神经元（避免在每个 batch 内重复调用 .any()）
        # Task 0 时 lifetime_elite_mask 全为 0，has_elite=False，跳过所有保护逻辑
        has_elite = bool(self.lifetime_elite_mask) and any(
            m.any().item() for m in self.lifetime_elite_mask.values()
        )

        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # --- Stream A: 显意识学习 ---
            student_logits = self.model(input_ids, attention_mask)
            loss_main = self.criterion(student_logits, labels)
            
            # --- Stream B: 梦境回放 ---
            loss_proto = torch.tensor(0.0, device=self.device)
            if prototype_memory is not None and len(prototype_memory.prototypes) > 0:
                proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=16)
                if proto_feats is not None:
                    proto_feats = proto_feats.to(self.device)
                    proto_labels = proto_labels.to(self.device)
                    if isinstance(self.model.classifier, nn.Sequential):
                        proto_logits = self.model.classifier(proto_feats)
                    else:
                        proto_logits = self.model.classifier(proto_feats)
                        
                    loss_proto = self.criterion(proto_logits, proto_labels) * proto_lambda

            # --- Stream C: 潜意识蒸馏 (提供软性直觉引导) ---
            loss_distill = torch.tensor(0.0, device=self.device)
            if frozen_model is not None:
                with torch.no_grad():
                    frozen_logits = frozen_model(input_ids, attention_mask)
                loss_distill = nn.MSELoss()(student_logits, frozen_logits) * distill_lambda

            total_loss_val = loss_main + loss_proto + loss_distill
            total_loss_val.backward()
            
            total_loss += total_loss_val.item()
            valid_batches += 1

            # [Optimization] Hebbian Accumulation
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None:
                        # ✅ [问题8修复] 累加求和替代 EMA，所有批次梯度等权重
                        # 旧 EMA (0.9*old + 0.1*new)：第1批权重=0.9^(N-1)→0，精英只由最后几批决定
                        # 新累加：task_importance = Σ|∂L/∂θ|，等价于 Taylor 重要性，学术标准做法
                        self.task_importance[name] += param.grad.abs()

            # 🔒 [Hard Subnetwork Masking] 绝对物理梯度隔离
            # 放弃有损的 SVD 近似，直接用终身精英二值掩码对梯度进行硬清零
            # mask=1 (精英): (1-1)=0 → 梯度强制清零，任何新任务的学习绝对无法侵入
            # mask=0 (非精英): (1-0)=1 → 梯度完整保留，新任务在此自由生长
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None and name in self.lifetime_elite_mask:
                        param.grad *= (1.0 - self.lifetime_elite_mask[name].to(param.device))

            # ✅ [Adam 泄漏修复] Pre-Step 快照 → Post-Step 精英还原
            # 根因：即使梯度被清零，Adam 中已积累的历史动量 m_t 不会立即归零
            #   m_t = β₁·m_{t-1} + (1-β₁)·0 = β₁·m_{t-1}  (指数衰减但非零!)
            #   optimizer.step() 仍会给精英神经元施加一个衰减中的残余位移
            # 修复策略：Step 前克隆精英权重 → Step 后将精英位置强制还原
            #   非精英位置：保留 Adam 的正常更新，不受影响
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
                            # 精英位置(mask=1)：强制还原 Step 前的值，完全消除 Adam 动量漂移
                            # 非精英位置(mask=0)：保留 Adam 正常更新
                            param.data.copy_(
                                mask * elite_snapshot[name] + (1.0 - mask) * param.data
                            )

            
            pbar.set_postfix({'Loss': f"{total_loss_val.item():.3f}"})

    def train_task(self, loader, prototype_memory, previous_mask):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr, 
            weight_decay=0.0 
        )
        
        # ✅ [双字典] 只清零当前任务积累器，终身精英掩码绝不触动
        for k in self.task_importance:
            self.task_importance[k].zero_()

        frozen_model = None

        if previous_mask is not None:
            print("🧠 [Distillation] 捕捉当前脑部快照，作为学习新任务的'潜意识模板'...")
            frozen_model = copy.deepcopy(self.model)
            frozen_model.eval()
            for param in frozen_model.parameters():
                param.requires_grad = False
            print("🔒 [Hard Masking] 梯度保护模式：终身精英神经元绝对锁定 (SVD-free)")

        print(f"🚀 [Trainer] Start waking training phase for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            self.train_epoch(
                loader, 
                prototype_memory, 
                previous_mask, 
                optimizer, 
                frozen_model=frozen_model,
                epoch_idx=epoch
            )
        
        # ✅ [双字典] 从本任务 task_importance 独立提取精英，永久固化入终身二值掩码
        # 关键：精英评选完全在本任务内部完成，T1 的梯度量级永远无法影响 T0 的精英资格
        # max 合并 0/1 掩码：一旦当过精英（=1），永不降级
        print("📚 [Lifetime Elite] 提取本任务精英神经元并永久固化 (0/1 binary union)...")
        with torch.no_grad():
            for k in self.task_importance:
                imp = self.task_importance[k]
                if imp.max() > 0:
                    # 分类器按行提取精英（行 = 类别节点），其余参数按元素提取
                    if 'classifier' in k and imp.dim() > 1:
                        row_imp = imp.mean(dim=1, keepdim=True)
                        threshold = torch.quantile(row_imp.float(), 0.85)
                        current_elite = (row_imp > threshold).float().expand_as(imp)
                    else:
                        threshold = torch.quantile(imp.float(), 0.85)
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

    def sleep(self, tokenizer, current_mask, prototype_memory, prototype_loader=None):
        if not self.args.use_sleep:
            return self.model, current_mask
            
        print(f"💤 [Trainer] Training done. Initiating Sleep Phase (Consolidation)...")
        
        config = {
            'alpha': self.args.alpha,
            'target_norm': self.args.target_norm,
            'no_rem': getattr(self.args, 'no_rem', False),
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
            prototype_loader=prototype_loader,   # ✅ [坐标系修复] NREM 后、REM 前刷新原型
        )
        
        if new_mask is not None:
            if current_mask is None:
                current_mask = new_mask
            else:
                for k, v in new_mask.items():
                    current_mask[k] = v 
            
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if total_params > 0:
                locked_params = sum(m.sum().item() for m in current_mask.values())
                print(f"🛡️  [Memory Capacity] Total Protected Neurons: {locked_params/total_params*100:.2f}%")
            
        return self.model, current_mask

    def evaluate(self, loader):
        self.model.eval()
        preds, gts = [], []
        
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                gts.extend(batch['labels'].cpu().numpy())
                
        from sklearn.metrics import accuracy_score
        return accuracy_score(gts, preds)