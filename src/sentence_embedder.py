from sentence_transformers import SentenceTransformer, models
import numpy as np

# Step 1: Load transformer (MiniLM)
word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 2: Replace pooling with max pooling
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=False,   # turn off mean pooling
    pooling_mode_max_tokens=True,     # turn on max pooling
    pooling_mode_cls_token=False      # optional: disable CLS if not needed
)

# Step 3: Create model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Step 4: Encode sample sentences
sentences = [
    "Machine learning is fascinating.",
    "Transformers are powerful models for NLP.",
    "I like pizza."
]
embeddings = model.encode(sentences)

# Step 5: View results
print("Sentence Embeddings:\n")
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5 values): {embedding[:5]}")
    print("-" * 60)
