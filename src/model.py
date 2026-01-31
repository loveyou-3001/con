import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoConfig
from modelscope.hub.snapshot_download import snapshot_download
from peft import get_peft_model, LoraConfig, TaskType

def get_bert_path(model_id='AI-ModelScope/bert-base-uncased'):
    """
    获取 BERT 模型路径，支持 ModelScope 缓存或本地/HF 路径
    """
    # 1. 尝试作为本地路径加载
    if os.path.exists(model_id):
        return model_id
        
    # 2. 尝试从 ModelScope 下载 (针对国内网络优化)
    cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
    try:
        # 如果 model_id 是短名 (e.g. bert-base-uncased)，尝试拼接前缀
        if '/' not in model_id and 'bert' in model_id:
            search_id = f'AI-ModelScope/{model_id}'
        else:
            search_id = model_id
            
        model_dir = os.path.join(cache_dir, search_id)
        if os.path.exists(model_dir):
            return model_dir
        else:
            return snapshot_download(search_id, cache_dir=cache_dir)
    except Exception as e:
        print(f"⚠️ ModelScope download failed: {e}. Falling back to HuggingFace ID: {model_id}")
        return model_id

class CosineLinear(nn.Module):
    """
    [Bio-Inspired Classifier]
    仿生余弦分类器：解耦 '突触强度'(Norm) 和 '突触模式'(Angle)。
    
    Why: 
    Sleep (NREM) 阶段会大幅压缩权重的 Norm (Synaptic Downscaling)。
    如果用普通 Linear，输出值会随 Norm 变小而剧烈下降，导致 Logits 坍塌。
    CosineLinear 即使 Norm 变小，只要方向(Angle)不变，分类结果依然准确。
    """
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Sigma 初始值设为 30.0
        # 较小的 sigma (如 1.0 或 10.0) 会导致 Softmax 后概率分布过于平坦 (Under-confidence)，
        # 梯度极小，导致模型无法收敛。
        if sigma:
            self.sigma = nn.Parameter(torch.tensor(30.0))
        else:
            self.register_buffer('sigma', torch.tensor(30.0))
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.in_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # 1. 归一化输入 (Batch, Hidden) -> (Batch, Hidden)
        out_norm = F.normalize(input, p=2, dim=1)
        
        # 2. 归一化权重 (Classes, Hidden) -> (Classes, Hidden)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 3. 计算余弦相似度并缩放
        # (Batch, Hidden) x (Hidden, Classes) = (Batch, Classes)
        return torch.mm(out_norm, w_norm.t()) * self.sigma

class HOPBertClassifier(nn.Module):
    def __init__(self, bert_path, num_classes, hop_order=2, use_lora=True, use_cosine=True):
        super(HOPBertClassifier, self).__init__()
        print(f"🏗️ Loading BERT from: {bert_path}")
        self.config = AutoConfig.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path, config=self.config)
        
        if use_lora:
            print("✨ Applying LoRA (Low-Rank Adaptation)...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=16,           # [建议] 稍微增大 Rank 以提升容量
                lora_alpha=32, 
                lora_dropout=0.1
            )
            self.bert = get_peft_model(self.bert, peft_config)
            
            # 打印可训练参数量
            self.bert.print_trainable_parameters()
        
        # ==============================================================================
        # 🔬 分类器选择 (支持消融实验)
        # use_cosine=True:  使用 CosineLinear (默认，抗 NREM 压缩)
        # use_cosine=False: 使用 nn.Linear (消融实验 A1)
        # ==============================================================================
        
        if not use_cosine:
            print("\n" + "="*80)
            print("⚠️⚠️ [ABLATION A1] w/o CosineLinear - Using nn.Linear ⚠️⚠️")
            print("⚠️ EXPECTATION: Accuracy may drop after NREM compression.")
            print("="*80 + "\n")
            
            # 使用普通全连接层 (bias=False 以控制变量)
            self.classifier = nn.Linear(self.config.hidden_size, num_classes, bias=False)
        else:
            print("\n✅ [Standard] Using Bio-Inspired CosineLinear (Orthogonal Synaptic Homeostasis).")
            self.classifier = CosineLinear(self.config.hidden_size, num_classes)
            
        # ==============================================================================

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            pooled_output = outputs['last_hidden_state'][:, 0, :] # CLS token
        else:
            pooled_output = outputs[0][:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits