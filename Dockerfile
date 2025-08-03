# Optimized LineLogic Docker Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/line48

# Install Python dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    supervision \
    opencv-python \
    numpy \
    pandas

# Copy project files
COPY src/ ./src/
COPY requirements.txt ./

# Create directories
RUN mkdir -p videos outputs logs models

# Download YOLO model
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Set working directory to src
WORKDIR /workspace/line48/src

# Default command
CMD ["python", "run_analysis.py", "--help"]