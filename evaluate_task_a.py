import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset_a import TaskADataset
from src.model import FetchMultiTaskModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = TaskADataset("data/task_a_test.json")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = FetchMultiTaskModel().to(device)
model.load_state_dict(torch.load("fetch_task_a_model.pth", map_location=device))
model.eval()

# Evaluation loop
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, task='A')
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Report
print("\n🧪 Evaluation Report:")
print(classification_report(
    all_labels, all_preds, target_names=["Camera Scan", "Email Upload", "Linked Account", "Manual Input"]
))
print("📊 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
