import torch
import torch.nn as nn
import numpy as np

def get_activation_variance(model, tokenizer, device, order=1):
    """
    [Sleep 核心]：输入高斯噪声/随机Token，通过 Hook 抓取 HOP 层的响应强度。
    假设：对随机噪声产生强烈稳定响应的路径，是已经固化的"核心骨架" (Gist)。
    """
    activations = {}
    
    def hook_fn(module, input, output):
        # output: [Batch, Dim * Order]
        activations['hop_output'] = output.detach()

    # 1. 注册 Hook 到 HOP 层
    # 注意：确保 hop_model.py 里层级名字是 model.hop
    handle = model.hop.register_forward_hook(hook_fn)
    
    # 2. 构造随机噪声输入 (模拟 REM 睡眠期的自发活动)
    model.eval()
    dummy_bs = 8
    seq_len = 32
    vocab_size = tokenizer.vocab_size
    
    # 随机生成 Token ID
    dummy_ids = torch.randint(0, vocab_size, (dummy_bs, seq_len)).to(device)
    dummy_mask = torch.ones(dummy_bs, seq_len).to(device)
    
    # 3. 前向传播 (No Grad)
    with torch.no_grad():
        model(dummy_ids, dummy_mask)
    
    handle.remove()
    return activations.get('hop_output', None)

def sleep_phase(model, tokenizer, device, config=None):
    """
    执行睡眠-突触缩减 (Synaptic Downscaling)
    """
    if config is None:
        # 默认配置：修剪稍微激进一点，看看效果
        config = {'alpha': 0.1, 'beta': 1.0, 'mask_threshold': 50}
        
    print(f"💤 进入睡眠阶段 (Sleep Phase)... 配置: {config}")
    
    alpha = config['alpha']          # 遗忘率 (模拟代谢，全局衰减)
    beta = config['beta']            # 保护率 (重要参数不降反升或保持)
    percentile = config['mask_threshold'] # 保护前百分之多少的参数 (Mask 稀疏度)
    
    # 1. 监测对噪声的响应
    hop_output = get_activation_variance(model, tokenizer, device)
    
    if hop_output is None:
        print("⚠️ Warning: Hook 没抓到数据，跳过 Sleep。")
        return model

    # 2. 计算重要性 Mask (Gist Extraction)
    # hop_output: [Batch, Feature_Dim] -> 取绝对值均值作为"活跃度"
    # 我们假设：在噪声输入下依然活跃的神经元，代表了模型的"固有偏置"或"结构性记忆"
    importance_score = hop_output.abs().mean(dim=0) # [Feature_Dim]
    
    # 计算阈值 (Top K)
    # 例如 percentile=20，意味着只有 Top 20% 的神经元连接被保护
    kth_value = torch.quantile(importance_score, 1 - percentile/100.0)
    
    # 生成 Mask: 1=重要(骨架), 0=噪音(需修剪)
    mask = (importance_score >= kth_value).float().to(device)
    sparsity = mask.mean().item()
    print(f"   >>> [Gist Extraction] 提取核心骨架: 保留了 {sparsity:.1%} 的强连接回路。")
    
    # 3. 执行突触缩减 (Synaptic Scaling)
    # 我们主要对 Classifier 的输入层权重进行修剪 (直接连接 HOP 特征的那一层)
    # 逻辑：LoRA 负责提取特征，MLP Head 负责组合特征。修剪 MLP Head 等于是修剪"概念的组合方式"。
    
    # model.classifier 是一个 Sequential
    # [0] 是 Linear(input_dim, hidden_dim)
    target_layer = model.classifier[0] 
    
    if isinstance(target_layer, nn.Linear):
        # 权重形状: [Out_Features, In_Features]
        # Mask 形状: [In_Features] -> 对应 HOP 的输出维度
        # 需要广播为 [1, In_Features] 以匹配权重矩阵的列
        mask_broadcast = mask.unsqueeze(0)
        
        with torch.no_grad():
            W = target_layer.weight.data
            original_norm = W.norm().item()
            
            # === 核心公式 (Bio-inspired Rule) ===
            # 对于被 Mask 选中的(重要): 权重 * (1 - alpha + beta) -> 甚至可能增强
            # 对于未被选中的(噪音): 权重 * (1 - alpha) -> 衰减
            # 这里我们设定 beta=alpha，意味着重要参数保持不变 (1-a+a=1)，噪音参数衰减
            
            # 如果 beta > alpha，则会有"记忆再巩固" (Reconsolidation) 的增强效果
            scaling_factor = (1 - alpha) + (mask_broadcast * beta)
            
            target_layer.weight.data *= scaling_factor
            
            new_norm = target_layer.weight.data.norm().item()
            print(f"   >>> [Synaptic Downscaling] 突触修剪完成。权重范数: {original_norm:.2f} -> {new_norm:.2f}")
            
    return model