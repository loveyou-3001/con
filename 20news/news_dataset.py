import json
import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=256):
        self.data = self.load_data(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentence']
        label = item['label']
        
        # 20News 只有一段文本，不需要 text_pair
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }