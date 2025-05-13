import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model import FetchMultiTaskModel
from src.dataset import TaskBDataset

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = TaskBDataset("data/task_b_samples.json")  # You can split into a test file later
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = FetchMultiTaskModel().to(device)
model.load_state_dict(torch.load("fetch_task_b_model.pth", map_location=device))
model.eval()

# Evaluation
true_scores = []
predicted_scores = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask, task='B')
        true_scores.extend(scores.cpu().numpy())
        predicted_scores.extend(preds.cpu().numpy())

# Metrics
mse = mean_squared_error(true_scores, predicted_scores)
mae = mean_absolute_error(true_scores, predicted_scores)

print(f"\nEvaluation on Test Set — MSE: {mse:.4f} | MAE: {mae:.4f}")
