import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset_a import TaskADataset
from src.model import FetchMultiTaskModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TaskADataset("data/task_a_samples.json")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = FetchMultiTaskModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# ✅ Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, task='A')
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    print("\n🧾 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[
        "Camera Scan", "Email Upload", "Linked Account", "Manual Input"
    ]))

    print("📊 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("-" * 60)


# ✅ Save the model
torch.save(model.state_dict(), "fetch_task_a_model.pth")
print("Model saved as fetch_task_a_model.pth")
