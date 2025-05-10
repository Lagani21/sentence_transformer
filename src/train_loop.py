import torch
import torch.nn as nn
import torch.optim as optim
from model import FetchMultiTaskModel

# --- CONFIGURATION ---
num_classes_task_a = 4
batch_size = 8
hidden_size = 384
epochs = 2

# --- INITIALIZE MODEL ---
model = FetchMultiTaskModel(num_classes_task_a=num_classes_task_a)
tokenizer = model.tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- LOSS FUNCTIONS & OPTIMIZER ---
loss_fn_task_a = nn.CrossEntropyLoss()
loss_fn_task_b = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# --- SIMULATE DUMMY DATA ---
# 50% classification, 50% regression
task_labels = ['A', 'B'] * 10  # 20 batches total

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch + 1}")

    for i, task in enumerate(task_labels):
        # --- Simulate input sentences ---
        sentences = [f"Sample sentence {j}" for j in range(batch_size)]
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

        if task == 'A':
            labels = torch.randint(0, num_classes_task_a, (batch_size,), device=device)
            logits = model(inputs['input_ids'], inputs['attention_mask'], task='A')
            loss = loss_fn_task_a(logits, labels)

            # Metric: Accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
            print(f"Batch {i+1:02d} [Task A] - Loss: {loss.item():.4f}, Acc: {acc.item():.2f}")

        elif task == 'B':
            targets = torch.rand(batch_size, device=device) * 5  # simulate scores from 0 to 5
            preds = model(inputs['input_ids'], inputs['attention_mask'], task='B')
            loss = loss_fn_task_b(preds, targets)

            # Metric: MSE (for logging only)
            mse = nn.functional.mse_loss(preds, targets)
            print(f"Batch {i+1:02d} [Task B] - Loss: {loss.item():.4f}, MSE: {mse.item():.2f}")

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Total Loss: {total_loss:.4f}")
