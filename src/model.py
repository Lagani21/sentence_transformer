# model.py
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

# Embedding model (max pooling)
def embedder():
    word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=True,
        pooling_mode_cls_token=False
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Multi-task model (classification + regression)
class FetchMultiTaskModel(nn.Module):
    def __init__(self, encoder_name='sentence-transformers/all-MiniLM-L6-v2', num_classes_task_a=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.task_a_head = nn.Linear(self.encoder.config.hidden_size, num_classes_task_a)
        self.task_b_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, task='A'):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        if task == 'A':
            return self.task_a_head(pooled_output)
        elif task == 'B':
            return self.task_b_head(pooled_output).squeeze(-1)
