import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from transformers import AutoTokenizer
from src.model import FetchMultiTaskModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FetchMultiTaskModel().to(device)
model.load_state_dict(torch.load("fetch_task_a_model.pth", map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Label map
label_map = {
    0: "Camera Scan",
    1: "Email Upload",
    2: "Linked Account",
    3: "Manual Input"
}

print("\nEnter a receipt description (type 'exit' to quit):\n")

while True:
    user_input = input("Your input: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    # Tokenize the single input
    encoded = tokenizer([user_input], padding=True, truncation=True, return_tensors="pt").to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], task='A')
        pred = torch.argmax(logits, dim=1).item()

    print(f"Predicted Receipt Type: {label_map[pred]}")
    print("-" * 50)
