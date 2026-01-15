import torch
import random

class PrototypeMemory:
    def __init__(self, num_classes, hidden_size, device):
        self.prototypes = {} 
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.device = device
        
    def update_prototypes(self, model, loader, device):
        model.eval()
        features_sum = {}
        features_count = {}
        
        # 使用 Hook 获取特征层
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0].detach() 
            return hook

        target_layer = model.classifier[0] if isinstance(model.classifier, torch.nn.Sequential) else model.classifier
        handle = target_layer.register_forward_hook(get_activation('feat'))
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                _ = model(input_ids, mask)
                
                if 'feat' in activation:
                    feats = activation['feat']
                    for i in range(len(labels)):
                        label = labels[i].item()
                        feat = feats[i]
                        if label not in features_sum:
                            features_sum[label] = torch.zeros_like(feat)
                            features_count[label] = 0
                        features_sum[label] += feat
                        features_count[label] += 1
        handle.remove()
        
        for label, total_feat in features_sum.items():
            mean_feat = total_feat / features_count[label]
            self.prototypes[label] = mean_feat.to(self.device)
        print(f"🧠 [Hippocampus] 原型记忆更新: {len(self.prototypes)} 个类别")
        torch.cuda.empty_cache()

    def get_prototype_batch(self, batch_size=32):
        if len(self.prototypes) == 0: return None, None
        sampled_labels = random.choices(list(self.prototypes.keys()), k=batch_size)
        batch_feats = torch.stack([self.prototypes[l] for l in sampled_labels])
        batch_labels = torch.tensor(sampled_labels).to(self.device)
        return batch_feats, batch_labels