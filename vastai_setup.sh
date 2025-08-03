#!/bin/bash

# VastAI kurulum scripti
echo "LineLogic VastAI Setup Starting..."

# Sistem güncellemesi
apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    wget \
    htop \
    nano

# Python sanal ortamı oluştur
python -m venv /workspace/line48_env
source /workspace/line48_env/bin/activate

# Python gereksinimlerini yükle
pip install --upgrade pip
pip install -r requirements.txt

# YOLO modelini önceden indir
echo "YOLO modelini indiriliyor..."
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Gerekli dizinleri oluştur
mkdir -p models
mkdir -p videos
mkdir -p outputs
mkdir -p logs

echo "Setup tamamlandı! Kullanım:"
echo "source /workspace/line48_env/bin/activate"
echo "cd /workspace/line48/src"
echo "python run_analysis.py --help"