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

        # 现在我们有了正交投影，蒸馏系数可以恢复到温和的 1.0
        proto_lambda = getattr(self.args, 'proto_lambda', 1.0)
        distill_lambda = getattr(self.args, 'distill_lambda', 1.0)

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
                        self.task_importance[name] = 0.9 * self.task_importance[name] + 0.1 * param.grad.abs()

            # 🌟 [Optimization] SVD 正交梯度投影 (Orthogonal Gradient Projection) 🌟
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None:
                        # 1. 针对 2D 参数矩阵 (LoRA, Classifier) 的高阶正交投影
                        if hasattr(self, 'orthogonal_bases') and name in self.orthogonal_bases:
                            P = self.orthogonal_bases[name]
                            grad_2d = param.grad.view(param.shape[0], -1).float()
                            # 矩阵乘法：强制抹除梯度在旧知识特征空间上的所有分量
                            proj_grad = torch.mm(grad_2d, P)
                            param.grad.copy_(proj_grad.view_as(param.grad))
                            
                        # 2. 针对 1D 参数 (如 Bias) 的降级元素掩码保护
                        elif previous_mask is not None and name in previous_mask:
                            mask = previous_mask[name]
                            param.grad *= (1.0 - mask)

            optimizer.step()
            
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
        self.orthogonal_bases = {} # 初始化正交矩阵字典
        
        if previous_mask is not None:
            print("🧠 [Distillation] 捕捉当前脑部快照，作为学习新任务的'潜意识模板'...")
            frozen_model = copy.deepcopy(self.model)
            frozen_model.eval()
            for param in frozen_model.parameters():
                param.requires_grad = False
                
            # 🌟 核心引擎：计算 Hebbian-SVD 投影矩阵
            print("📐 [Orthogonal] 计算 Hebbian-SVD 梯度正交投影矩阵 (保护核心特征子空间)...")
            for name, param in self.model.named_parameters():
                # 只对 2D 以上的矩阵进行 SVD 分解
                # ✅ [双字典] 使用 lifetime_elite_mask（绝对 0/1 二值图）
                # 所有历史任务的精英神经元地位平等，SVD 只看拓扑结构，不受量级影响
                if param.requires_grad and param.dim() > 1 and name in self.lifetime_elite_mask:
                    imp_mat = self.lifetime_elite_mask[name].view(param.shape[0], -1).float()
                    
                    if imp_mat.sum() == 0:
                        continue
                        
                    try:
                        # 奇异值分解
                        U, S, Vh = torch.linalg.svd(imp_mat, full_matrices=False)
                        
                        # 截断策略：提取包含 90% 能量的主成分方向
                        total_energy = (S ** 2).sum()
                        current_energy = 0
                        k = 0
                        for i in range(len(S)):
                            current_energy += S[i] ** 2
                            if current_energy / total_energy > 0.90:
                                k = i + 1
                                break
                                
                        if k > 0:
                            V_k = Vh[:k, :].T  # [in_dim, k]
                            # P = I - V * V^T (构建正交零空间)
                            I = torch.eye(imp_mat.shape[1], device=self.device)
                            P = I - torch.mm(V_k, V_k.T)
                            self.orthogonal_bases[name] = P
                    except RuntimeError as e:
                        # 防止 SVD 在极端情况下不收敛
                        print(f"   ⚠️ SVD failed for {name}, fallback to default masking.")
                        pass

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

    def sleep(self, tokenizer, current_mask, prototype_memory):
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
            self.lifetime_elite_mask,   # ✅ [双字典] 传入绝对 0/1 精英掩码，sleep 无需任何 quantile 计算
            prototype_memory,
            previous_mask=current_mask 
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