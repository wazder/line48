#!/bin/bash

# Optimized LineLogic Setup Script
echo "🚀 LineLogic Optimized Setup Starting..."

# Install core dependencies only
pip install --upgrade pip
pip install ultralytics supervision opencv-python numpy pandas torch torchvision

# Test GPU availability
echo "🔥 Testing GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download YOLO model
echo "📥 Downloading YOLO model..."
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Create necessary directories
mkdir -p videos outputs logs models

echo "✅ Optimized setup completed!"
echo ""
echo "🚀 Quick start:"
echo "cd src"
echo "python run_analysis.py --video '../videos/your_video.mp4' --frame-logic"
echo ""
echo "📊 For SAM functionality, install additional dependencies:"
echo "pip install segment-anything matplotlib Pillow scipy"