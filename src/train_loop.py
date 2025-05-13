#Task 4
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from torch.utils.data import DataLoader
from dataset import TaskADataset, TaskBDataset
from model import FetchMultiTaskModel
import torch.nn as nn
import torch
from tqdm import tqdm

#Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FetchMultiTaskModel().to(device)

# Load datasets
task_a_loader = DataLoader(TaskADataset("data/task_a_samples.json"), batch_size=4, shuffle=True)
task_b_loader = DataLoader(TaskBDataset("data/task_b_samples.json"), batch_size=4, shuffle=True)

# Loss functions
loss_fn_a = nn.CrossEntropyLoss()
loss_fn_b = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Multi-Task Training Loop
epochs = 5

for epoch in range(epochs):
    model.train()
    task_b_iter = iter(task_b_loader)

    # 1. Initialize metric tracking
    total_a_loss = 0
    total_a_correct = 0
    total_a_samples = 0
    all_a_preds = []
    all_a_labels = []

    total_b_loss = 0
    true_b_scores = []
    pred_b_scores = []

    for batch_a in tqdm(task_a_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        # Task A (classification)
        input_ids_a = batch_a['input_ids'].to(device)
        attention_mask_a = batch_a['attention_mask'].to(device)
        labels_a = batch_a['label'].to(device)

        logits_a = model(input_ids=input_ids_a, attention_mask=attention_mask_a, task='A')
        loss_a = loss_fn_a(logits_a, labels_a)

        #2. Track Task A metrics
        preds_a = torch.argmax(logits_a, dim=1)
        total_a_correct += (preds_a == labels_a).sum().item()
        total_a_samples += labels_a.size(0)
        total_a_loss += loss_a.item()
        all_a_preds.extend(preds_a.cpu().numpy())
        all_a_labels.extend(labels_a.cpu().numpy())

        #Task B (regression)
        try:
            batch_b = next(task_b_iter)
        except StopIteration:
            task_b_iter = iter(task_b_loader)
            batch_b = next(task_b_iter)

        input_ids_b = batch_b['input_ids'].to(device)
        attention_mask_b = batch_b['attention_mask'].to(device)
        scores_b = batch_b['score'].to(device)

        preds_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b, task='B')
        loss_b = loss_fn_b(preds_b, scores_b)

        #Track Task B metrics
        total_b_loss += loss_b.item()
        true_b_scores.extend(scores_b.cpu().numpy())
        pred_b_scores.extend(preds_b.detach().cpu().numpy())

        #Combined loss 
        loss = loss_a + loss_b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 3. After loop: evaluate & print metrics
    from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

    avg_a_loss = total_a_loss / len(task_a_loader)
    a_accuracy = total_a_correct / total_a_samples

    avg_b_loss = total_b_loss / len(task_b_loader)
    b_mse = mean_squared_error(true_b_scores, pred_b_scores)
    b_mae = mean_absolute_error(true_b_scores, pred_b_scores)

    print(f"\n Epoch {epoch+1} — Task A (Classification)")
    print(f"Loss: {avg_a_loss:.4f} | Accuracy: {a_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(all_a_labels, all_a_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_a_labels, all_a_preds))

    print(f"\n Epoch {epoch+1} — Task B (Regression)")
    print(f"Loss: {avg_b_loss:.4f} | MSE: {b_mse:.4f} | MAE: {b_mae:.4f}")
    print("-" * 60)