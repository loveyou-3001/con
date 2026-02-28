import torch
import torch.nn as nn
import math

def synaptic_downscaling(model, importance_matrix, previous_mask, config, prototype_memory, device):
    """
    Sleep-HOP 睡眠机制核心：
    包含 NREM (容量释放) 和 REM (虚拟满血复活与边界修复)
    """
    print("🌙 [Sleep] 进入睡眠周期 (NREM + REM)...")
    
    alpha = config.get('alpha', 0.1)      
    base_target_norm = config.get('target_norm', 10.0)
    global_masks = {} 
    
    num_classes = len(prototype_memory.prototypes) if prototype_memory else 15
    
    # ==========================================
    # 阶段 1：NREM (慢波睡眠) - 物理压缩
    # ==========================================
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            
            if 'sigma' in name:
                global_masks[name] = torch.ones_like(param)
                continue

            if 'lora' in name:
                soft_alpha = config.get('lora_alpha', 0.01)
                if soft_alpha == 0:
                    global_masks[name] = torch.ones_like(param)
                    continue
                
                if name in importance_matrix:
                    imp = importance_matrix[name]
                    lora_imp_norm = imp / (imp.max() + 1e-8) if imp.max() > 0 else torch.zeros_like(param)
                else:
                    lora_imp_norm = torch.zeros_like(param)
                
                decay = 1.0 - soft_alpha * (1.0 - lora_imp_norm)
                param.data *= decay
                global_masks[name] = torch.ones_like(param) * (1.0 - soft_alpha)
                continue

            if name in importance_matrix:
                imp = importance_matrix[name]
                imp_norm = imp / (imp.max() + 1e-8) if imp.max() > 0 else imp
            else:
                imp_norm = torch.zeros_like(param)

            # 核心：解耦模长与方向 (按行缩放)
            if 'classifier' in name and param.dim() > 1:
                row_imp = imp_norm.mean(dim=1, keepdim=True)
                threshold = torch.quantile(row_imp, 0.85) 
                row_mask = (row_imp > threshold).float()
                global_masks[name] = row_mask.expand_as(param) 
                
                decay = 1.0 - alpha * (1.0 - row_imp)
                
                if previous_mask is not None and name in previous_mask:
                    prev_mask = previous_mask[name].to(param.device)
                    row_prev_mask = prev_mask.max(dim=1, keepdim=True)[0]
                    decay = decay * (1.0 - row_prev_mask) + 1.0 * row_prev_mask
                    global_masks[name] = torch.max(global_masks[name], prev_mask)
                
                param.data *= decay

            else:
                threshold = torch.quantile(imp_norm, 0.85) 
                current_mask = (imp_norm > threshold).float()
                global_masks[name] = current_mask
                
                decay = 1.0 - alpha * (1.0 - imp_norm)
                
                if previous_mask is not None and name in previous_mask:
                    prev_mask = previous_mask[name].to(param.device)
                    decay = decay * (1.0 - prev_mask) + 1.0 * prev_mask
                    global_masks[name] = torch.max(global_masks[name], prev_mask)
                
                param.data *= decay
            
            if param.dim() > 1:
                curr_norm = param.norm()
                dynamic_target_norm = base_target_norm * math.sqrt(max(1, num_classes) / 15.0)
                if curr_norm > dynamic_target_norm:
                    scaling = dynamic_target_norm / (curr_norm + 1e-8)
                    param.data *= scaling

    print("   ✅ [NREM] 突触缩减与容量释放完成.")

    # ==========================================
    # 阶段 2：REM (快速眼动) - 虚拟满血复活与微调
    # ==========================================
    if not config.get('use_rem', True) or num_classes == 0:
        return global_masks

    print(f"   💭 [REM] 开始做梦微调... (Classes: {num_classes})")

    # 🌟 核心机制：保存真实内存尺寸，并临时拉伸到 1.0 防止梯度爆炸
    saved_norms = {}
    with torch.no_grad():
        for name, param in model.classifier.named_parameters():
            if 'weight' in name and param.dim() > 1:
                norm = param.norm(dim=1, keepdim=True)
                saved_norms[name] = norm.clone()
                # 临时拔高到单位球面
                param.data /= (norm + 1e-8)

    params_to_opt = []
    for name, param in model.classifier.named_parameters():
        if 'sigma' in name:
            param.requires_grad = False 
        else:
            params_to_opt.append(param)
            
    if not params_to_opt:
        return global_masks

    optimizer = torch.optim.AdamW(params_to_opt, lr=1e-2, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    dream_cycles = max(600, min(1500, num_classes * 10))

    for i in range(dream_cycles):
        proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=32)
        if proto_feats is None: break
        
        proto_feats = proto_feats.to(device)
        proto_labels = proto_labels.to(device)
        
        noise = torch.randn_like(proto_feats) * 0.1
        noisy_feats = proto_feats + noise
        
        optimizer.zero_grad()
        outputs = model.classifier(noisy_feats)
            
        loss = criterion(outputs, proto_labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(params_to_opt, max_norm=1.0)
        optimizer.step()
        
    # 🌟 梦醒时分：完美的角度装回被压缩的物理内存中
    with torch.no_grad():
        for name, param in model.classifier.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # 剔除 Adam 带来的冗余长度变化
                current_norm = param.norm(dim=1, keepdim=True)
                param.data /= (current_norm + 1e-8)
                # 完璧归赵：恢复 NREM 的物理压缩比例
                param.data *= saved_norms[name]

    for name, param in model.classifier.named_parameters():
        if 'sigma' in name:
            param.requires_grad = True
            
    print("   ✅ [REM] 记忆再巩固完成 (虚拟复活机制执行完毕).")
    
    return global_masks