import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.sleep import synaptic_downscaling # 确保与修复后的函数名一致

class HOPTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        
        # 1. 重要性矩阵 (Hebbian Trace) 初始化
        self.importance_matrix = {}
        
        # 2. [Optimization] 预先缓存可训练参数，提升大循环效率
        self.trainable_params_cache = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 初始重要性为极小值而非纯零，防止除零错误
                self.importance_matrix[name] = torch.full_like(param.data, 1e-8)
                self.trainable_params_cache.append((name, param))

    def train_epoch(self, loader, prototype_memory, previous_mask, optimizer, epoch_idx=0):
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        # 读取回放强度
        proto_lambda = getattr(self.args, 'proto_lambda', 2.0)

        # 进度条配置
        pbar = tqdm(loader, desc=f"   Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # --- Stream A: 新任务学习 (Wake Phase) ---
            logits = self.model(input_ids, attention_mask)
            loss_main = self.criterion(logits, labels)
            
            # --- Stream B: 旧知识原型回放 (Wake Replay) ---
            loss_proto = torch.tensor(0.0, device=self.device)
            if prototype_memory is not None and len(prototype_memory.prototypes) > 0:
                # 加上 self. 前缀
                proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=self.args.batch_size // 2)
                
                if proto_feats is not None:
                    # 直接喂给分类器，绕过 BERT/LoRA，速度提升千倍
                    proto_logits = self.model.classifier(proto_feats.to(self.device))
                    loss_proto = self.criterion(proto_logits, proto_labels.to(self.device)) * proto_lambda

            # 总 Loss 计算
            total_loss_val = loss_main + loss_proto
            total_loss_val.backward()
            
            # --- 核心机制 1: Hebbian 重要性累积 ---
            # 公式: $I_t = \gamma I_{t-1} + (1-\gamma) |Grad|$
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None:
                        self.importance_matrix[name] = 0.9 * self.importance_matrix[name] + 0.1 * param.grad.abs()

            # --- 核心机制 2: 梯度屏蔽 (Gradient Masking) ---
            # 阻止梯度修改已被 Sleep 保护的旧知识神经元
            if previous_mask is not None:
                for name, param in self.trainable_params_cache:
                    if param.grad is not None and name in previous_mask:
                        mask = previous_mask[name].to(self.device)
                        param.grad *= (1.0 - mask)

            optimizer.step()
            
            total_loss += total_loss_val.item()
            valid_batches += 1
            pbar.set_postfix({'loss': f"{total_loss_val.item():.4f}"})
            
        return total_loss / valid_batches if valid_batches > 0 else 0

    def train_task(self, loader, prototype_memory, previous_mask):
        """执行单个任务的完整训练循环"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr, 
            weight_decay=0.01 
        )
        
        # 任务开始前，衰减旧任务的重要性权重，为新知识腾挪注意力
        for k in self.importance_matrix:
            self.importance_matrix[k] *= 0.1 

        print(f"☀️ [Trainer] 启动任务训练 (Wake Phase)...")
        for epoch in range(self.args.epochs):
            avg_loss = self.train_epoch(loader, prototype_memory, previous_mask, optimizer, epoch_idx=epoch)
            if (epoch + 1) % 5 == 0:
                print(f"   > Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            
        torch.cuda.empty_cache()
        gc.collect()

    def sleep(self, tokenizer, current_mask, prototype_memory):
        """
        调用核心睡眠机制：
        1. NREM 执行物理压缩与容量回收
        2. REM 执行梦境复习与边界修复
        """
        if not getattr(self.args, 'use_sleep', False):
            return self.model, current_mask
            
        print(f"💤 [Trainer] 任务训练结束，触发睡眠巩固 (NREM + REM)...")
        
        # 包装消融实验配置
        config = {
            'alpha': self.args.alpha,
            'target_norm': self.args.target_norm,
            'use_rem': not getattr(self.args, 'no_rem', False),
            'lora_alpha': getattr(self.args, 'lora_alpha', 0.01),
        }
        
        # 执行物理级压缩
        new_mask = synaptic_downscaling(
            self.model, 
            self.importance_matrix, 
            current_mask, 
            config, 
            prototype_memory,
            self.device
        )
        
        # --- 掩码合并逻辑 (Union of Protection) ---
        if new_mask is not None:
            if current_mask is None:
                current_mask = new_mask
            else:
                for k, v in new_mask.items():
                    # 只要该位置被保护过，就终身保护
                    current_mask[k] = torch.max(current_mask[k], v.to(current_mask[k].device))
            
            # 打印容量健康统计
            self._log_capacity_stats(current_mask)
            
        return self.model, current_mask

    def _log_capacity_stats(self, mask_dict):
        """分析并打印模型容量占用情况"""
        total_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        locked_p = sum(m.sum().item() for m in mask_dict.values())
        
        # 统计压缩率 (分类器中模长接近 0 的参数)
        with torch.no_grad():
            low_norm_p = 0
            cls_total = 0
            for name, param in self.model.classifier.named_parameters():
                if 'weight' in name:
                    low_norm_p += (param.data.abs() < 1e-3).sum().item()
                    cls_total += param.numel()

        print(f"🛡️  [Capacity] 保护神经元占比: {locked_p/total_p*100:.2f}%")
        if cls_total > 0:
            print(f"📉 [NREM] 分类器冗余压缩率: {low_norm_p/cls_total*100:.2f}%")

    def evaluate(self, loader):
        """标准的评估逻辑"""
        self.model.eval()
        all_preds, all_gts = [], []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, mask)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_gts.extend(labels.cpu().numpy())
                
        return accuracy_score(all_gts, all_preds)