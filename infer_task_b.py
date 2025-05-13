import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import torch
from transformers import AutoTokenizer
from src.model import FetchMultiTaskModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FetchMultiTaskModel().to(device)
model.load_state_dict(torch.load("fetch_task_b_model.pth", map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

print("\nType a user complaint (type 'exit' to quit):\n")

while True:
    text = input("Support message: ").strip()
    if text.lower() in ["exit", "quit"]:
        print("Done.")
        break

    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        pred = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], task='B')
        pred_score = round(pred.item(), 2)

    print(f"Predicted Frustration Score: {pred_score}")
    print("-" * 50)
