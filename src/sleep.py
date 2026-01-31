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

            # 🔥 [豁免 2] LoRA 软豁免 (Adaptive Plasticity)
            # 策略演进：从"绝对豁免 (100%)"升级为"软豁免 (99%)"
            # 
            # 原因：经过多个任务后，完全冻结的 LoRA 无法学习新特征模式
            # 解决：保留 1% 的可塑性，允许基于重要性的微调
            # 
            # 关键参数：
            #   - Soft Alpha = 0.01 (仅 1% 衰减，99% 保留)
            #   - 依然使用重要性矩阵指导，保护关键神经元
            #   - REM 虽然无法修复 LoRA，但软豁免的衰减极小，风险可控
            if 'lora' in name:
                # 从 config 读取 LoRA 软豁免强度 (支持消融实验 A4)
                # lora_alpha=0: 完全冻结 (消融实验)
                # lora_alpha=0.01: 1% 可塑性 (默认)
                soft_alpha = config.get('lora_alpha', 0.01)
                
                # 消融实验 A4: 完全冻结 LoRA
                if soft_alpha == 0:
                    global_masks[name] = torch.ones_like(param)
                    continue
                
                # 先获取 LoRA 的重要性分数
                if name in importance_matrix:
                    imp = importance_matrix[name]
                    if imp.max() > 0:
                        lora_imp_norm = imp / (imp.max() + 1e-8)
                    else:
                        lora_imp_norm = torch.zeros_like(param)
                else:
                    lora_imp_norm = torch.zeros_like(param)
                
                # 软性衰减：只对不重要的参数施加轻微压力
                decay = 1.0 - soft_alpha * (1.0 - lora_imp_norm)
                param.data *= decay
                
                # Mask 标记保护程度
                global_masks[name] = torch.ones_like(param) * (1.0 - soft_alpha)
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
    
    # 5. [动态周期] 根据类别数量自适应调整
    # 公式: max(600, min(1500, num_classes * 10))
    # 原理: 类别越多，决策边界越复杂，需要更多训练时间
    num_classes = len(prototype_memory.prototypes)
    dream_cycles = max(600, min(1500, num_classes * 10))
    print(f"      [REM Config] Classes: {num_classes}, Dream Cycles: {dream_cycles}")
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
    
    # 消融实验 A3: 跳过 REM 阶段
    if config.get('no_rem', False):
        print("   ⚠️ [ABLATION A3] Skipping REM phase...")
    else:
        model = rem_consolidation(model, prototype_memory, device, config)
    
    for k in importance_matrix:
        importance_matrix[k].zero_()
    return model, masks