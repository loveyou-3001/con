import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim import SGD

def synaptic_downscaling(model, importance_matrix, config, previous_mask):
    """
    [NREM Phase] 慢波睡眠：执行突触稳态缩减 (Synaptic Homeostasis)
    
    关键机制：
    1. 计算当前任务的重要性 (Importance)。
    2. 生成高阈值 Mask (Top 85%)，锁定大部分知识。
    3. 执行衰减 (Downscaling)，但对 '历史核心参数' (previous_mask) 进行强制豁免。
    """
    print("   💤 [NREM] 进入慢波睡眠 (SWS): 执行 Synaptic Downscaling...")
    
    alpha = config.get('alpha', 0.1)     
    target_norm = config.get('target_norm', 10.0)
    
    global_masks = {} 
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            # 只处理可训练参数 (LoRA + Head)
            if not param.requires_grad: continue
            
            # 1. 获取并归一化重要性
            if name in importance_matrix:
                imp = importance_matrix[name]
                if imp.max() > 0:
                    imp_norm = imp / (imp.max() + 1e-8)
                else:
                    imp_norm = imp
            else:
                imp_norm = torch.zeros_like(param)

            # 2. 生成当前任务的新 Mask
            # [关键] 提高阈值到 0.85 (保护 85% 的参数)
            # LoRA 参数本来就少，必须采取激进的保护策略，防止特征漂移
            threshold = torch.quantile(imp_norm, 0.85) 
            current_mask = (imp_norm > threshold).float()
            global_masks[name] = current_mask
            
            # 3. 计算衰减系数 (Soft Decay)
            # 基础逻辑：重要性越低，衰减越狠 (decay < 1.0)
            decay = 1.0 - alpha * (1.0 - imp_norm)
            
            # 4. [资产冻结机制] 历史保护
            # 如果参数在之前的任务中已经被标记为核心 (previous_mask=1)，则强制不衰减
            if previous_mask is not None and name in previous_mask:
                prev_mask = previous_mask[name].to(param.device)
                
                # 逻辑混合：
                # 如果 prev_mask=1 -> decay=1.0 (保护)
                # 如果 prev_mask=0 -> decay=decay (按当前重要性衰减)
                decay = decay * (1.0 - prev_mask) + 1.0 * prev_mask
                
                # 更新当前输出的 Mask，确保这个保护状态传递给下一个任务
                global_masks[name] = torch.max(global_masks[name], prev_mask)
            
            # 5. 执行物理缩减
            param.data *= decay
            
            # 6. 稳态归一化 (Renormalization)
            # 防止某些参数因为反复保护而无限膨胀
            curr_norm = param.norm()
            if curr_norm > target_norm:
                scaling = target_norm / (curr_norm + 1e-8)
                param.data *= scaling
                
    return model, global_masks

def rem_consolidation(model, prototype_memory, device, config):
    """
    [REM Phase] 快速眼动睡眠：梦境回放 (Dreaming)
    
    由于 Prototype 存储的是特征 (Features)，无法反向传播更新 BERT Encoder。
    因此 REM 阶段专注于利用记忆原型快速微调分类器 (Classifier Alignment)。
    """
    if prototype_memory is None or len(prototype_memory.prototypes) == 0:
        return model

    print("   👁️ [REM] 进入快速眼动睡眠: 正在做梦 (Classifier Replay)...")
    
    # 切换到训练模式
    model.train()
    
    # 锁定 Encoder，只训练 Classifier
    # 这一步是为了防止在特征输入下，优化器误更新了不需要梯度的部分（虽然没梯度也更不了，但为了保险）
    if isinstance(model.classifier, nn.Sequential):
        params_to_opt = model.classifier.parameters()
    else:
        params_to_opt = model.classifier.parameters()
        
    # 使用较大的学习率 (LR=0.01) 进行快速对齐
    optimizer = SGD(params_to_opt, lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 增加梦境循环次数
    dream_cycles = 100 
    
    for _ in range(dream_cycles):
        # 获取梦境片段 (Prototypes)
        proto_feats, proto_labels = prototype_memory.get_prototype_batch(batch_size=32)
        if proto_feats is None: break
        
        proto_feats = proto_feats.to(device)
        proto_labels = proto_labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 (只过分类器)
        if isinstance(model.classifier, nn.Sequential):
            outputs = model.classifier(proto_feats)
        else:
            outputs = model.classifier(proto_feats)
            
        loss = criterion(outputs, proto_labels)
        loss.backward()
        optimizer.step()
        
    print("   ✅ [REM] 记忆再巩固完成。")
    return model

def sleep_phase(model, tokenizer, device, config, importance_matrix, prototype_memory, previous_mask):
    """
    睡眠主入口
    """
    # 1. NREM: 物理缩减 (带历史保护)
    model, masks = synaptic_downscaling(model, importance_matrix, config, previous_mask)
    
    # 2. REM: 功能重组
    model = rem_consolidation(model, prototype_memory, device, config)
    
    # 3. 清空当天的海马体 (重要性矩阵清零，准备迎接新的一天)
    for k in importance_matrix:
        importance_matrix[k].zero_()
        
    return model, masks