FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg && rm -rf /var/lib/apt/lists/*

# Install CUDA PyTorch (adjust CUDA version as needed)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]
