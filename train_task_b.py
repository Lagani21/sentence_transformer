import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dataset import TaskBDataset
from src.model import FetchMultiTaskModel

# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TaskBDataset("data/task_b_samples.json")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = FetchMultiTaskModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

# ✅ Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    true_scores = []
    predicted_scores = []

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask, task='B')
        loss = loss_fn(preds, scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        true_scores.extend(scores.cpu().numpy())
        predicted_scores.extend(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    mse = mean_squared_error(true_scores, predicted_scores)
    mae = mean_absolute_error(true_scores, predicted_scores)

    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")

# ✅ Save model
torch.save(model.state_dict(), "fetch_task_b_model.pth")
print("Model saved as fetch_task_b_model.pth")
