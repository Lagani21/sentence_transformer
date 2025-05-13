import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

import torch
from transformers import AutoTokenizer
from src.model import FetchMultiTaskModel

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FetchMultiTaskModel().to(device)
model.load_state_dict(torch.load("fetch_task_b_model.pth", map_location=device))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

print("\nEnter a receipt description (type 'exit' to quit):\n")

while True:
    user_input = input("Your input: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting. Goodbye!")
        break

    # Tokenize input
    encoded = tokenizer([user_input], padding=True, truncation=True, return_tensors="pt").to(device)

    # Predict
    with torch.no_grad():
        score = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], task='B')
        score = score.item()

    print(f"Predicted Score: {score:.4f}")
    print("-" * 50)
