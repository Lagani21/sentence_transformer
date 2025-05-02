from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample input sentences
sentences = [
    "Machine learning is fascinating.",
    "Transformers are powerful models for NLP.",
    "I enjoy working on deep learning projects."
]

# Generate fixed-length sentence embeddings
embeddings = model.encode(sentences)

# Print results
print("Sentence Embeddings:\n")
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5 values): {embedding[:5]}")
    print("-" * 60)
