#!/bin/bash

# VastAI SAM + LineLogic Setup Script
echo "🎯 SAM + LineLogic VastAI Setup Starting..."

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
python -m venv /workspace/sam_line48_env
source /workspace/sam_line48_env/bin/activate

# Python gereksinimlerini yükle
pip install --upgrade pip

# SAM ve diğer gereksinimler
echo "📦 Installing SAM and dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segment-anything
pip install ultralytics supervision opencv-python numpy matplotlib Pillow scipy pandas

echo "🔥 Testing PyTorch CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# SAM modellerini indir
echo "📥 Downloading SAM models..."
mkdir -p /workspace/sam_models
cd /workspace/sam_models

# SAM ViT-B (fastest, 375MB)
if [ ! -f "sam_vit_b_01ec64.pth" ]; then
    echo "Downloading SAM ViT-B model..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# SAM ViT-L (better quality, 1.25GB) - optional
# if [ ! -f "sam_vit_l_0b3195.pth" ]; then
#     echo "Downloading SAM ViT-L model..."
#     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# fi

# SAM ViT-H (best quality, 2.56GB) - optional for RTX 4090
# if [ ! -f "sam_vit_h_4b8939.pth" ]; then
#     echo "Downloading SAM ViT-H model..."
#     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# fi

cd /workspace/line48

# YOLO modelini önceden indir
echo "📥 Downloading YOLO model..."
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Gerekli dizinleri oluştur
mkdir -p models videos outputs logs

# SAM model path'lerini src dizinine kopyala
ln -sf /workspace/sam_models/*.pth src/

echo "✅ SAM + LineLogic setup completed!"
echo ""
echo "🚀 Usage:"
echo "source /workspace/sam_line48_env/bin/activate"
echo "cd /workspace/line48/src"
echo ""
echo "# Basic SAM analysis"
echo "python run_sam_analysis.py --video ../videos/your_video.mp4 --sam-model vit_b"
echo ""
echo "# High quality with ViT-H (if downloaded)"
echo "python run_sam_analysis.py --video ../videos/your_video.mp4 --sam-model vit_h"
echo ""
echo "# Fast processing (skip frames)"
echo "python run_sam_analysis.py --video ../videos/your_video.mp4 --sam-model vit_b --skip-frames 2"