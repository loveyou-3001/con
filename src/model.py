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
    if os.path.exists(model_id):
        return model_id
        
    cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
    try:
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
    
    原理: 
    Sleep (NREM) 阶段会大幅压缩权重的 Norm (释放突触容量)。
    由于我们在 forward 时进行动态归一化 (Dynamic Normalization)，
    被 NREM 物理压缩的旧任务特征在计算打分时，会被重新拉回单位球面。
    这保证了新旧任务之间的“绝对公平竞争”，彻底消除了灾难性遗忘！
    """
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Sigma 控制输出的置信度 (Temperature)
        if sigma:
            self.sigma = nn.Parameter(torch.tensor(30.0))
        else:
            self.register_buffer('sigma', torch.tensor(30.0))
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.in_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # 🌟 优化 1: 加入 eps=1e-8，防止 NREM 极端压缩导致的除零错误 (NaN)
        out_norm = F.normalize(input, p=2, dim=1, eps=1e-8)
        
        # 🌟 核心修复: 动态归一化权重
        # 无论底层的 self.weight 被 NREM 压缩到了多小 (哪怕是 0.001)，
        # 这里的 w_norm 永远是 1.0。这就切断了物理容量和分类打分之间的联系。
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-8)
        
        # 🌟 优化 2: 使用 F.linear 代替 torch.mm
        # F.linear 原生执行 input @ weight.T，计算图更健壮，显存分配更优
        cosine_sim = F.linear(out_norm, w_norm)
        
        return self.sigma * cosine_sim

class HOPBertClassifier(nn.Module):
    def __init__(self, bert_path, num_classes, hop_order=2, use_lora=True, use_cosine=True, lora_rank=16):
        super(HOPBertClassifier, self).__init__()
        print(f"🏗️ Loading BERT from: {bert_path}")
        self.config = AutoConfig.from_pretrained(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path, config=self.config)
        
        if use_lora:
            print(f"✨ Applying LoRA (rank={lora_rank})...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                inference_mode=False, 
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.1
            )
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()
        
        # ==============================================================================
        # 🔬 分类器选择 (支持消融实验)
        # ==============================================================================
        if not use_cosine:
            print("\n" + "="*80)
            print("⚠️⚠️ [ABLATION A1] w/o CosineLinear - Using nn.Linear ⚠️⚠️")
            print("⚠️ EXPECTATION: Accuracy will drop heavily after NREM compression.")
            print("="*80 + "\n")
            
            self.classifier = nn.Linear(self.config.hidden_size, num_classes, bias=False)
        else:
            print("\n✅ [Standard] Using Bio-Inspired CosineLinear (Orthogonal Synaptic Homeostasis).")
            self.classifier = CosineLinear(self.config.hidden_size, num_classes)
            
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