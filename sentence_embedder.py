#Task 1: Sentence Embedding

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from src.model import embedder

embedder = embedder()
sentences = [
    "Machine learning is fascinating.",
    "Transformers are powerful models for NLP.",
    "I like pizza."
]

embeddings = embedder.encode(sentences)

print("== Max Pooling Sentence Embeddings ==")
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    print("-" * 60)
