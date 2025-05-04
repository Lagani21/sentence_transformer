import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class FetchMultiTaskModel(nn.Module):
    def __init__(self, encoder_name='sentence-transformers/all-MiniLM-L6-v2', num_classes_task_a=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.task_a_head = nn.Linear(self.encoder.config.hidden_size, num_classes_task_a)
        self.task_b_head = nn.Linear(self.encoder.config.hidden_size, 1)  # For regression

    def forward(self, input_ids, attention_mask, task='A'):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        if task == 'A':
            return self.task_a_head(pooled_output)
        elif task == 'B':
            return self.task_b_head(pooled_output)
