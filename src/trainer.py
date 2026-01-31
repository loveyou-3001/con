import torch
import torch.nn as nn
import gc
from tqdm import tqdm  # [新增] 引入进度条库
from src.sleep import sleep_phase

class HOPTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        
        # Hebbian Trace 初始化
        self.importance_matrix = {}
        
        # [Optimization] 预先缓存可训练参数
        # 避免在 train_epoch 的每个 batch 中重复调用 named_parameters() 产生的 CPU 开销
        self.trainable_params_cache = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance_matrix[name] = torch.zeros_like(param.data)
                self.trainable_params_cache.append((name, param))

    def train_epoch(self, loader, prototype_memory, previous_mask, optimizer, epoch_idx=0):
        self.model.train()
        total_loss = 0
        valid_batches = 0

        # [Replay Strategy] 混合复习系数
        proto_lambda = getattr(self.args, 'proto_lambda', 1.0)

        # [新增] 使用 tqdm 包装 loader，显示进度条
        # desc 显示当前是第几个 Epoch
        pbar = tqdm(loader, desc=f"  Epoch {epoch_idx+1}/{self.args.epochs}", leave=False)

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # --- Stream A: 新任务学习 ---
            student_logits = self.model(input_ids, attention_mask)
            loss_main = self.criterion(student_logits, labels)
            
            # --- Stream B: 旧知识复习 (Wake Replay) ---
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

            # 总 Loss
            total_loss_val = loss_main + loss_proto
            total_loss_val.backward()
            
            total_loss += total_loss_val.item()
            valid_batches += 1

            # [Optimization] 优化的 Hebbian Accumulation
            with torch.no_grad():
                for name, param in self.trainable_params_cache:
                    if param.grad is not None:
                        self.importance_matrix[name] = 0.9 * self.importance_matrix[name] + 0.1 * param.grad.abs()

            # [Optimization] 优化的 Gradient Masking (Protection)
            if previous_mask is not None:
                for name, param in self.trainable_params_cache:
                    if param.grad is not None and name in previous_mask:
                        mask = previous_mask[name]
                        param.grad *= (1.0 - mask)

            optimizer.step()
            
            # [新增] 实时更新进度条后缀，显示当前 Loss
            pbar.set_postfix({'loss': f"{total_loss_val.item():.4f}"})
            
        # Epoch 结束后的汇总打印
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            # 可以选择打印或只依赖 tqdm
            # print(f"    📉 Avg Loss: {avg_loss:.4f}")

    def train_task(self, loader, prototype_memory, previous_mask):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr, 
            weight_decay=0.0 
        )
        
        # 衰减旧重要性
        for k in self.importance_matrix:
            self.importance_matrix[k] *= 0.1 

        print(f"🚀 [Trainer] Start waking training phase for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            # 传入 epoch 索引以更新进度条描述
            self.train_epoch(loader, prototype_memory, previous_mask, optimizer, epoch_idx=epoch)
            
        torch.cuda.empty_cache()
        gc.collect()

    def sleep(self, tokenizer, current_mask, prototype_memory):
        if not self.args.use_sleep:
            return self.model, current_mask
            
        print(f"💤 [Trainer] Training done. Initiating Sleep Phase (Consolidation)...")
        
        config = {
            'alpha': self.args.alpha,
            'target_norm': self.args.target_norm,
            # 消融实验参数
            'no_rem': getattr(self.args, 'no_rem', False),
            'lora_alpha': getattr(self.args, 'lora_alpha', 0.01),
        }
        
        self.model, new_mask = sleep_phase(
            self.model, 
            tokenizer, 
            self.device, 
            config, 
            self.importance_matrix, 
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
        
        # [可选] 评估也可以加进度条，如果数据很多的话
        # pbar = tqdm(loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                gts.extend(batch['labels'].cpu().numpy())
                
        from sklearn.metrics import accuracy_score
        return accuracy_score(gts, preds)