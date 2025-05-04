import torch.nn as nn
from sentence_transformers import SentenceTransformer

class FetchMultiTaskModel(nn.Module):
    def __init__(self, encoder_name='all-MiniLM-L6-v2', num_classes_task_a=4):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        self.task_a_head = nn.Linear(384, num_classes_task_a)  # Classification head
        self.task_b_head = nn.Linear(384, 1)                    # Regression head (for frustration score)

    def forward(self, sentences, task='A'):
        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)

        if task == 'A':
            return self.task_a_head(embeddings)  # Output shape: [batch_size, 4]
        elif task == 'B':
            return self.task_b_head(embeddings)  # Output shape: [batch_size, 1]
