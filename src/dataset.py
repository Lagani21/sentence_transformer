import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TaskADataset(Dataset):
    def __init__(self, json_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        
        self.label_map = {
            "Camera Scan": 0,
            "Email Upload": 1,
            "Linked Account": 2,
            "Manual Input": 3
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        label = self.label_map[item['label']]

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
class TaskBDataset(Dataset):
    def __init__(self, json_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        score = item['score']  # float between 0.0 and 1.0

        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float)
        }
