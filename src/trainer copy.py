import torch
import torch.nn as nn
import gc
from src.sleep import sleep_phase

class HOPTrainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        
        # Hebbian Trace
        self.importance_matrix = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance_matrix[name] = torch.zeros_like(param.data)

    def train_epoch(self, loader, prototype_memory, previous_mask, optimizer):
        self.model.train()
        total_loss = 0
        valid_batches = 0

        # [Replay Strategy] 混合复习系数
        # 如果有记忆，loss = loss_new + proto_lambda * loss_old
        proto_lambda = getattr(self.args, 'proto_lambda', 1.0)

        for batch in loader:
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
                # 获取原型 batch (特征或原始数据)
                # 注意：memory.py 中存储的是 feature (768维)
                # 所以我们只能过 classifier，不能过 BERT
                proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=16)
                
                if proto_feats is not None:
                    proto_feats = proto_feats.to(self.device)
                    proto_labels = proto_labels.to(self.device)
                    
                    # 关键：我们需要一个能直接接受 features 的 forward 接口
                    # 假设 model.classifier 可以直接处理 features
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

            # [Mechanism] Hebbian Accumulation
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # 累积梯度的绝对值
                        self.importance_matrix[name] = 0.9 * self.importance_matrix[name] + 0.1 * param.grad.abs()

            # [Mechanism] Gradient Masking (Protection)
            if previous_mask is not None:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None and name in previous_mask:
                        mask = previous_mask[name]
                        # 强力阻断：受保护参数不更新
                        param.grad *= (1.0 - mask)

            optimizer.step()
            
        if valid_batches > 0:
            print(f"    📉 Avg Loss: {total_loss / valid_batches:.4f}")

    def train_task(self, loader, prototype_memory, previous_mask):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr, 
            weight_decay=0.0 
        )
        
        # 衰减旧重要性，为新任务腾出记录空间
        for k in self.importance_matrix:
            self.importance_matrix[k] *= 0.1 

        print(f"🚀 [Trainer] Start waking training phase for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            self.train_epoch(loader, prototype_memory, previous_mask, optimizer)
            
        torch.cuda.empty_cache()
        gc.collect()

    def sleep(self, tokenizer, current_mask, prototype_memory):
        if not self.args.use_sleep:
            return self.model, current_mask
            
        print(f"💤 [Trainer] Training done. Initiating Sleep Phase (Consolidation)...")
        
        config = {
            'alpha': self.args.alpha,
            'target_norm': self.args.target_norm
        }
        
        # 传递 current_mask 以实现累积保护
        self.model, new_mask = sleep_phase(
            self.model, 
            tokenizer, 
            self.device, 
            config, 
            self.importance_matrix, 
            prototype_memory,
            previous_mask=current_mask 
        )
        
        # 更新 Mask (并集策略)
        if new_mask is not None:
            if current_mask is None:
                current_mask = new_mask
            else:
                for k, v in new_mask.items():
                    current_mask[k] = v # sleep.py 已经做了 merge 逻辑，这里直接赋值
            
            # 打印保护比例
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