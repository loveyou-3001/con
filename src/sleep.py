import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim import AdamW  # 使用更稳健的 AdamW

def synaptic_downscaling(model, importance_matrix, config, previous_mask):
    """
    [NREM Phase] 慢波睡眠：执行突触稳态缩减
    修复：显式豁免 'sigma' 参数，防止尺度因子坍塌。
    """
    print("   💤 [NREM] 进入慢波睡眠 (SWS): 执行 Synaptic Downscaling...")
    
    alpha = config.get('alpha', 0.1)     
    target_norm = config.get('target_norm', 10.0)
    
    global_masks = {} 
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            
            # 🔥 [稳健性修复 1] 绝对豁免 Sigma
            # Sigma 控制分类器的"体温" (logits scale)，不应参与突触的"剪枝"。
            if 'sigma' in name:
                global_masks[name] = torch.ones_like(param)
                continue

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
            
            # 6. 稳态归一化
            if param.dim() > 1:
                curr_norm = param.norm()
                if curr_norm > target_norm:
                    scaling = target_norm / (curr_norm + 1e-8)
                    param.data *= scaling
                
    return model, global_masks

def rem_consolidation(model, prototype_memory, device, config):
    """
    [REM Phase] 快速眼动睡眠：梦境回放
    修复：使用 AdamW + Label Smoothing + Sigma Freezing 实现稳健微调。
    """
    if prototype_memory is None or len(prototype_memory.prototypes) == 0:
        return model

    print("   👁️ [REM] 进入快速眼动睡眠: 正在做梦 (Robust Mode)...")
    
    model.train()
    
    # 🔥 [稳健性修复 2] 只优化权重，冻结 Sigma
    # 我们希望 REM 阶段去修复权重的"方向"(Angle)，而不是通过调整 Sigma 来 cheat loss。
    params_to_opt = []
    for name, param in model.classifier.named_parameters():
        if 'sigma' in name:
            param.requires_grad = False # 暂时冻结
        else:
            params_to_opt.append(param)
            
    # 🔥 [稳健性修复 3] 使用 AdamW 替代 SGD
    # AdamW 能自动适应 NREM 缩减后变小的权重尺度，避免梯度爆炸。
    # lr=1e-3 是标准值，不激进。
    optimizer = AdamW(params_to_opt, lr=1e-3, weight_decay=1e-4)
    
    # 🔥 [稳健性修复 4] 标签平滑 (Label Smoothing)
    # 防止模型对 Prototype 过拟合 (Loss=0)，保持梯度持续流动以打磨边界。
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    dream_cycles = 150 # 适度增加循环次数
    noise_std = 0.1    # 温和的噪声
    
    for i in range(dream_cycles):
        proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=32)
        if proto_feats is None: break
        
        proto_feats = proto_feats.to(device)
        proto_labels = proto_labels.to(device)
        
        # 注入噪声增强鲁棒性
        noise = torch.randn_like(proto_feats) * noise_std
        noisy_feats = proto_feats + noise
        
        optimizer.zero_grad()
        
        # Forward (Classifier only)
        if isinstance(model.classifier, nn.Sequential):
            outputs = model.classifier(noisy_feats)
        else:
            outputs = model.classifier(noisy_feats)
            
        loss = criterion(outputs, proto_labels)
        loss.backward()
        
        # 梯度裁剪保险
        nn.utils.clip_grad_norm_(params_to_opt, max_norm=1.0)
        
        optimizer.step()
        
    # 恢复 Sigma 的梯度状态 (为下一次清醒期做准备)
    for name, param in model.classifier.named_parameters():
        if 'sigma' in name:
            param.requires_grad = True
            
    print("   ✅ [REM] 记忆再巩固完成 (Stable).")
    return model

def sleep_phase(model, tokenizer, device, config, importance_matrix, prototype_memory, previous_mask):
    model, masks = synaptic_downscaling(model, importance_matrix, config, previous_mask)
    model = rem_consolidation(model, prototype_memory, device, config)
    for k in importance_matrix:
        importance_matrix[k].zero_()
    return model, masks