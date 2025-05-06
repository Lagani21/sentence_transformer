import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import TaskADataset
from torch.utils.data import DataLoader


dataset = TaskADataset("data/task_a_samples.json")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    print(batch['input_ids'].shape, batch['label'])  # Sanity check
    break
 