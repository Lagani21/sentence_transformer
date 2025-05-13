# ✅ Use slim Python image (small & ARM compatible)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git gcc

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install Python packages (includes PyTorch, Transformers, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Run training loop
CMD ["python", "src/train_loop.py"]
