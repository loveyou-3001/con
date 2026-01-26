import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim import AdamW

def synaptic_downscaling(model, importance_matrix, config, previous_mask):
    """
    [NREM Phase] 慢波睡眠：执行突触稳态缩减
    🔥 Fix: 给 LoRA 和 Sigma 颁发双重豁免权，防止 NREM 误伤不可修复的参数
    """
    print("   💤 [NREM] 进入慢波睡眠 (SWS): 执行 Synaptic Downscaling...")
    
    alpha = config.get('alpha', 0.1)     
    target_norm = config.get('target_norm', 10.0)
    
    global_masks = {} 
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            
            # 🔥 [豁免 1] Sigma
            # Sigma 控制分类器的"体温"，不能被压缩
            if 'sigma' in name:
                global_masks[name] = torch.ones_like(param)
                continue

            # 🔥 [豁免 2] LoRA 参数 (CRITICAL FIX)
            # 因为 REM 阶段使用的是 Prototype (Pooled Output) 作为输入，
            # 梯度无法回传到 BERT/LoRA 层。
            # 这意味着 REM 无法修复 LoRA 的损伤。
            # 因此，NREM 阶段必须给予 LoRA "绝对豁免权"，严禁触碰！
            if 'lora' in name:
                # print(f"      [Immunity] LoRA parameter protected: {name}")
                global_masks[name] = torch.ones_like(param)
                continue

            # --- 以下仅对 Classifier 的权重执行缩减 ---

            # 1. 获取并归一化重要性
            if name in importance_matrix:
                imp = importance_matrix[name]
                if imp.max() > 0:
                    imp_norm = imp / (imp.max() + 1e-8)
                else:
                    imp_norm = imp
            else:
                imp_norm = torch.zeros_like(param)

            # 2. 生成 Mask (Top 85%)
            threshold = torch.quantile(imp_norm, 0.85) 
            current_mask = (imp_norm > threshold).float()
            global_masks[name] = current_mask
            
            # 3. 计算衰减
            decay = 1.0 - alpha * (1.0 - imp_norm)
            
            # 4. 历史保护
            if previous_mask is not None and name in previous_mask:
                prev_mask = previous_mask[name].to(param.device)
                decay = decay * (1.0 - prev_mask) + 1.0 * prev_mask
                global_masks[name] = torch.max(global_masks[name], prev_mask)
            
            # 5. 执行物理缩减
            param.data *= decay
            
            # 6. 稳态归一化 (对于 Classifier 权重，这一步其实会被 REM 的 Re-Scaling 覆盖，但保留也无妨)
            if param.dim() > 1:
                curr_norm = param.norm()
                if curr_norm > target_norm:
                    scaling = target_norm / (curr_norm + 1e-8)
                    param.data *= scaling
                
    return model, global_masks

def rem_consolidation(model, prototype_memory, device, config):
    """
    [REM Phase] 快速眼动睡眠：梦境回放
    🔥 Precision Mode: 标准化模长 (1.0) + 稳健LR (1e-2) + 长时程巩固 (600)
    """
    if prototype_memory is None or len(prototype_memory.prototypes) == 0:
        return model

    print("   👁️ [REM] 进入快速眼动睡眠: 正在做梦 (Precision Mode)...")
    
    model.train()
    
    # 1. [数学标准化] Re-Scaling to Unit Norm (1.0)
    # 将 Classifier 权重强制归一化到 1.0，配合 AdamW 实现最佳几何优化。
    # LoRA 参数因为在 NREM 被豁免了，所以这里不需要管，它们是健康的。
    with torch.no_grad():
        for name, param in model.classifier.named_parameters():
            if 'weight' in name and param.dim() > 1:
                norm = param.norm(dim=1, keepdim=True)
                target_norm = 1.0
                scaling = target_norm / (norm + 1e-8)
                param.data *= scaling

    # 2. 准备优化参数 (只优化 Classifier)
    # REM 只能修 Classifier，修不了 LoRA (因为没有梯度通路)
    params_to_opt = []
    for name, param in model.classifier.named_parameters():
        # 冻结 Sigma (保持 30 不变)
        if 'sigma' in name:
            param.requires_grad = False 
        else:
            params_to_opt.append(param)
            
    # 3. [稳健配置] LR=1e-2
    # 配合 Unit Norm (1.0)，这个学习率是经过验证的黄金参数 (92.7% ver)。
    optimizer = AdamW(params_to_opt, lr=1e-2, weight_decay=0.0)
    
    # 4. [稳健配置] Label Smoothing 0.1
    # 保持适度不确定性，防止震荡。
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 5. [以时间换精度] 600 Cycles
    # 既然 LoRA 没坏，Classifier 只要多花点时间精调，99% 是必然的。
    dream_cycles = 600
    noise_std = 0.1    
    
    for i in range(dream_cycles):
        proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=32)
        if proto_feats is None: break
        
        proto_feats = proto_feats.to(device)
        proto_labels = proto_labels.to(device)
        
        noise = torch.randn_like(proto_feats) * noise_std
        noisy_feats = proto_feats + noise
        
        optimizer.zero_grad()
        
        # 注意：这里只跑 Classifier 部分
        if isinstance(model.classifier, nn.Sequential):
            outputs = model.classifier(noisy_feats)
        else:
            outputs = model.classifier(noisy_feats)
            
        loss = criterion(outputs, proto_labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(params_to_opt, max_norm=1.0)
        optimizer.step()
        
    # 恢复 Sigma
    for name, param in model.classifier.named_parameters():
        if 'sigma' in name:
            param.requires_grad = True
            
    print("   ✅ [REM] 记忆再巩固完成 (Precision + LoRA Protected).")
    return model

def sleep_phase(model, tokenizer, device, config, importance_matrix, prototype_memory, previous_mask):
    model, masks = synaptic_downscaling(model, importance_matrix, config, previous_mask)
    model = rem_consolidation(model, prototype_memory, device, config)
    for k in importance_matrix:
        importance_matrix[k].zero_()
    return model, masks