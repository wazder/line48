# VastAI deployment için PyTorch ve CUDA destekli base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Sistem paketlerini güncelle ve gerekli araçları yükle
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /workspace/line48

# Python gereksinimlerini kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Model dizinini oluştur
RUN mkdir -p models

# YOLO modelini indir (isteğe bağlı - ilk çalıştırmada otomatik indirilir)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Çalışma dizinini src'ye ayarla
WORKDIR /workspace/line48/src

# Default komut
CMD ["python", "run_analysis.py", "--help"]