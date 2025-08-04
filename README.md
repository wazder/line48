# LineLogic - Object Tracking and Line Crossing Detection

A computer vision system for detecting and counting objects crossing virtual lines in video streams using YOLO models and ByteTrack tracking with frame-based validation.

## 🎯 Features

- **Object Detection**: YOLO-based detection of people, backpacks, handbags, and suitcases
- **Line Crossing Detection**: Tracks objects crossing virtual lines with direction detection
- **Frame-based Validation**: Filters out brief, unreliable detections
- **Command-line Interface**: Flexible parameter configuration
- **GPU Accelerated**: Optimized for CUDA-capable devices
- **SAM Integration**: Optional Segment Anything Model for pixel-perfect segmentation

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/wazder/line48.git
cd line48

# Run optimized setup
chmod +x setup_optimized.sh
./setup_optimized.sh
```

### 2. Basic Usage
```bash
cd src

# Process video with frame-based logic
python run_analysis.py --video "../videos/your_video.mp4" --frame-logic

# Custom parameters for RTX 4090
python run_analysis.py \
  --video "../videos/your_video.mp4" \
  --frame-logic \
  --confidence 0.25 \
  --iou 0.45 \
  --imgsz 1280

# First 200 frames SAM analysis
python run_first_200.py
```

### 3. SAM Enhanced Analysis (Optional)
```bash
# Install SAM dependencies
pip install segment-anything matplotlib Pillow scipy

# Run SAM analysis
python run_sam_analysis.py \
  --video "../videos/your_video.mp4" \
  --sam-model vit_b \
  --download-sam
```

## 📊 Current Performance
- **Person**: 95% accuracy
- **Backpack**: 79% accuracy  
- **Handbag**: 44% accuracy
- **Suitcase**: 85% accuracy

## 🛠️ Dependencies

**Core Requirements:**
- ultralytics>=8.0.0
- supervision>=0.18.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- pandas>=1.5.0
- torch>=2.0.0
- torchvision>=0.15.0

**SAM Optional:**
- segment-anything>=1.0
- matplotlib>=3.6.0
- Pillow>=9.0.0
- scipy>=1.9.0

## 📁 Output Files

- **Processed Video**: `outputs/video_processed_timestamp.mp4`
- **Detection Log**: `logs/video_log_timestamp.csv`
- **Results Summary**: `logs/video_results_timestamp.csv`

## ⚙️ Configuration

### Command-Line Parameters
```bash
--confidence 0.25      # YOLO confidence threshold
--iou 0.45            # NMS IoU threshold
--imgsz 1280          # Input image size
--frame-logic         # Enable frame-based validation
--device cuda:0       # GPU device
```

### Frame Logic Parameters
```bash
--min-safe-time 0.5        # Minimum time for safe predictions
--min-uncertain-time 0.28  # Minimum time for uncertain predictions
--min-very-brief-time 0.17 # Minimum time for very brief predictions
```

## 🎬 VastAI Deployment

```bash
# Quick VastAI setup
cd /workspace
git clone https://github.com/wazder/line48.git
cd line48
./setup_optimized.sh

# Download video from Google Drive
gdown "YOUR_GOOGLE_DRIVE_LINK" -O videos/input_video.mp4

# Run analysis
cd src
python run_analysis.py --video "../videos/input_video.mp4" --frame-logic
```

See [VASTAI_DEPLOYMENT.md](VASTAI_DEPLOYMENT.md) for detailed deployment instructions.

## 🔬 SAM Integration

For higher accuracy with pixel-perfect segmentation:

```bash
# Install SAM dependencies
pip install segment-anything matplotlib Pillow scipy

# Run SAM analysis
python run_sam_analysis.py --video "../videos/video.mp4" --sam-model vit_b
```

See [SAM_LINELOGIC_GUIDE.md](SAM_LINELOGIC_GUIDE.md) for detailed SAM usage.

## 📈 Analysis Tools

The `analysis_tools/` directory contains scripts for:
- Comparing results against ground truth
- Analyzing missed detections  
- Performance optimization
- Confidence distribution analysis

## 🐳 Docker Usage

```bash
# Build and run
docker build -t linelogic .
docker run --gpus all -it --rm \
  -v $(pwd)/videos:/workspace/line48/videos \
  -v $(pwd)/outputs:/workspace/line48/outputs \
  linelogic
```

## 📋 Project Structure

```
line48/
├── src/                    # Main source code
│   ├── run_analysis.py     # Main analysis runner
│   ├── run_sam_analysis.py # SAM-enhanced analysis
│   ├── run_first_200.py    # SAM testing (first 200 frames)
│   ├── config.py           # Main configuration settings
│   ├── line_config.py      # Interactive line placement
│   ├── frame_logic.py      # Frame-based validation
│   ├── sam_utils.py        # SAM model utilities
│   ├── sam_frame_logic.py  # SAM segment tracking
│   ├── video_utils.py      # Video selection utilities
│   ├── frame_overlay.py    # Frame annotation system
│   ├── results_exporter.py # CSV export utilities
│   └── utils.py            # Core utility functions
├── analysis_tools/         # Analysis and debugging scripts
├── videos/                 # Input video files
├── outputs/                # Processed video outputs
└── logs/                   # Detection logs and results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.