FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 11.7 version)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Set workdir
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]
