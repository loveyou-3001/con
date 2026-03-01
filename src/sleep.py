import torch
import torch.nn as nn
import math

def sleep_phase(model, tokenizer, device, config, importance_matrix, prototype_memory, previous_mask):
    """
    Sleep-HOP 睡眠机制核心 (入口函数)：
    包含 NREM (物理容量释放) 和 REM (虚拟满血复活与边界精修)
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
            
            # Sigma 绝对豁免
            if 'sigma' in name:
                global_masks[name] = torch.ones_like(param)
                continue

            # ✅ [方案X] LoRA 参数取消无差别豆免，走重要性分位数筛选流程
            # 旧代码用 global_masks = ones * (1 - 0.01) = 0.99 全员保护
            # 导致仅两个任务后即锁死 88%+ 空间，后续任务无法学习
            # 现在：LoRA 和普通层一样，只保护重要性 > 85th 百分位数的神经元

            # 常规参数重要性获取
            if name in importance_matrix:
                imp = importance_matrix[name]
                imp_norm = imp / (imp.max() + 1e-8) if imp.max() > 0 else imp
            else:
                imp_norm = torch.zeros_like(param)

            # 🌟 核心：分类器权重的解耦压缩 (按行等比例缩放)
            if 'classifier' in name and param.dim() > 1:
                row_imp = imp_norm.mean(dim=1, keepdim=True)
                threshold = torch.quantile(row_imp, 0.85) 
                row_mask = (row_imp > threshold).float()
                global_masks[name] = row_mask.expand_as(param) 
                
                decay = 1.0 - alpha * (1.0 - row_imp)
                
                # 历史保护叠加
                if previous_mask is not None and name in previous_mask:
                    prev_mask = previous_mask[name].to(param.device)
                    row_prev_mask = prev_mask.max(dim=1, keepdim=True)[0]
                    decay = decay * (1.0 - row_prev_mask) + 1.0 * row_prev_mask
                    global_masks[name] = torch.max(global_masks[name], prev_mask)
                
                param.data *= decay

            else:
                # 其他常规层（按元素压缩）
                threshold = torch.quantile(imp_norm, 0.85) 
                current_mask = (imp_norm > threshold).float()
                global_masks[name] = current_mask
                
                decay = 1.0 - alpha * (1.0 - imp_norm)
                
                if previous_mask is not None and name in previous_mask:
                    prev_mask = previous_mask[name].to(param.device)
                    decay = decay * (1.0 - prev_mask) + 1.0 * prev_mask
                    global_masks[name] = torch.max(global_masks[name], prev_mask)
                
                param.data *= decay
            
            # 动态容量天花板
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
        # 🚨 已经删除了此处的清零代码
        return model, global_masks

    print(f"   💭 [REM] 开始做梦微调... (Classes: {num_classes})")

    # 🌟 记住真实内存尺寸，并临时拔高到单位球面
    saved_norms = {}
    with torch.no_grad():
        for name, param in model.classifier.named_parameters():
            if 'weight' in name and param.dim() > 1:
                norm = param.norm(dim=1, keepdim=True)
                saved_norms[name] = norm.clone()
                param.data /= (norm + 1e-8)

    params_to_opt = []
    for name, param in model.classifier.named_parameters():
        if 'sigma' in name:
            param.requires_grad = False 
        else:
            params_to_opt.append(param)
            
    if params_to_opt:
        optimizer = torch.optim.AdamW(params_to_opt, lr=1e-2, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        dream_cycles = max(600, min(1500, num_classes * 10))

        for i in range(dream_cycles):
            proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=32)
            if proto_feats is None: break
            
            proto_feats = proto_feats.to(device)
            proto_labels = proto_labels.to(device)
            
            # 🌟 降低噪音，精准微雕
            noise = torch.randn_like(proto_feats) * 0.05
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
                    current_norm = param.norm(dim=1, keepdim=True)
                    param.data /= (current_norm + 1e-8)
                    param.data *= saved_norms[name]

        for name, param in model.classifier.named_parameters():
            if 'sigma' in name:
                param.requires_grad = True
                
    print("   ✅ [REM] 记忆再巩固完成 (虚拟复活机制执行完毕).")
    
    # 🚨 已经删除了末尾处的清空 Hebbian Trace 代码
        
    return model, global_masks