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

# Label map (same as training)
label_map = {
    0: "Camera Scan",
    1: "Email Upload",
    2: "Linked Account",
    3: "Manual Input"
}

# Input sentence(s)
sentences = [
    "Scanned my Walmart receipt using the app",
    "Uploaded Uber Eats receipt from Gmail",
    "Entered receipt details manually in the form"
]

# Tokenize
encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

# Predict
with torch.no_grad():
    logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], task='A')
    preds = torch.argmax(logits, dim=1)

# Output results
for sentence, pred in zip(sentences, preds):
    print(f"📄 Sentence: {sentence}")
    print(f"🎯 Predicted Receipt Type: {label_map[pred.item()]}")
    print("-" * 60)
