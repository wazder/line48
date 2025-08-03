# LineLogic - Object Tracking and Line Crossing Detection System

## Project Overview

This is a computer vision system designed to detect and count objects crossing virtual lines in video streams. The system was developed to track specific objects (people, backpacks, handbags, and suitcases) as they cross predefined virtual lines in video footage.

### What We're Trying to Achieve

The main goal is to create an automated system that can:
1. **Detect objects** in video streams using YOLO models
2. **Track objects** across frames using ByteTrack
3. **Count crossings** when objects pass through virtual lines
4. **Validate detections** using frame-based logic to reduce false positives
5. **Generate reports** with detailed statistics and analysis

### Target Use Cases
- **Security monitoring** - Counting people entering/exiting areas
- **Retail analytics** - Tracking customer flow and bag detection
- **Transportation** - Monitoring passenger and luggage movement
- **Event management** - Crowd flow analysis

## Key Features

- **Multi-object detection**: People, backpacks, handbags, and suitcases
- **Virtual line crossing**: Configurable line placement and detection
- **Frame-based validation**: Filters out brief, unreliable detections
- **Interactive video selection**: Easy video browsing and selection
- **Video cropping**: Interactive crop selection with preview
- **Comprehensive logging**: Detailed CSV outputs for analysis
- **Performance evaluation**: Tools to compare against ground truth
- **Command-line interface**: Flexible parameter configuration

## Current Performance

Based on testing with sample videos:
- **Person detection**: ~95% accuracy
- **Backpack detection**: ~79% accuracy  
- **Handbag detection**: ~44% accuracy (main challenge area)
- **Suitcase detection**: ~85% accuracy

## Quick Setup Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

The system uses YOLO models for object detection. You'll need to download them:

```bash
# Create models directory
mkdir models

# Download YOLO models (you can choose one or multiple)
# YOLOv8x (largest, most accurate)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O models/yolov8x.pt

# YOLOv11n (fastest, smallest)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt -O models/yolov11n.pt

# YOLOv11x (balanced)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11x.pt -O models/yolov11x.pt
```

### 3. Prepare Your Videos

Place your video files in the `videos/` directory. The system supports common formats (MP4, MOV, AVI, etc.).

## Usage Examples

### Basic Usage

```bash
cd src
python run_analysis.py
```

This will:
1. Show available videos
2. Let you select a video interactively
3. Run the analysis with default settings
4. Generate output videos and CSV logs

### Advanced Usage

```bash
# Process specific video with custom parameters
python run_analysis.py --video "../videos/your_video.mp4" --confidence 0.3 --iou 0.4 --imgsz 1280

# Use frame-based validation (recommended)
python run_analysis.py --video "../videos/your_video.mp4" --frame-logic

# Crop video first, then analyze
python run_analysis.py --crop-video "../videos/your_video.mp4"
```

### Command Line Options

**Model Parameters:**
- `--confidence`: Detection confidence threshold (default: 0.25)
- `--iou`: NMS IoU threshold (default: 0.45)
- `--imgsz`: Input image size (default: 1024)

**Logic Parameters:**
- `--frame-logic`: Use frame-based validation (default)
- `--basic-logic`: Use basic tracking logic
- `--min-safe-time`: Minimum time for safe predictions (default: 0.5s)
- `--min-uncertain-time`: Minimum time for uncertain predictions (default: 0.28s)

**Utility Options:**
- `--list-videos`: Show available videos
- `--crop-video`: Crop video with interactive selection
- `--help`: Show all available options

## Project Structure

```
LineLogic/
├── src/                    # Main source code
│   ├── run_analysis.py     # Main analysis runner
│   ├── logic.py           # Basic tracking logic
│   ├── frame_logic.py     # Frame-based validation
│   ├── config.py          # Configuration settings
│   ├── utils.py           # Utility functions
│   ├── video_config.py    # Video selection utilities
│   ├── video_cropper.py   # Video cropping functionality
│   ├── results_exporter.py # Results export utilities
│   └── line_config.py     # Line configuration
├── analysis_tools/         # Analysis and debugging scripts
├── models/                 # YOLO model files (download separately)
├── videos/                 # Input video files (add your videos here)
├── outputs/                # Processed video outputs (created automatically)
├── logs/                   # Detection logs (created automatically)
└── requirements.txt        # Python dependencies
```

## Configuration

### Line Configuration

Edit `src/line_config.py` to customize line positions:

```python
# Example line configuration
LINES = [
    {
        'name': 'Entry Line',
        'start': (100, 200),
        'end': (500, 200),
        'direction': 'up'  # or 'down', 'left', 'right'
    }
]
```

### Model Configuration

Edit `src/config.py` to adjust default parameters:

```python
# Default model settings
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMG_SIZE = 1024
```

## Output Files

The system generates several output files:

1. **Processed Video**: `outputs/[video_name]_processed_[timestamp].mp4`
   - Video with detection overlays and counts

2. **Detection Log**: `logs/[video_name]_log_[timestamp].csv`
   - Detailed frame-by-frame detection data

3. **Results Summary**: `logs/[video_name]_results_[timestamp].csv`
   - Summary statistics and analysis

4. **Cropped Videos**: `videos/cropped_videos/[video_name]_cropped.mp4`
   - Cropped versions of input videos

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The system works on CPU, but GPU acceleration requires proper CUDA setup
2. **Model Download**: Ensure models are downloaded to the `models/` directory
3. **Video Format**: Some video formats may require FFmpeg installation
4. **Memory Issues**: Reduce `--imgsz` parameter for lower memory usage

### Performance Tips

- Use `yolov11n.pt` for faster processing
- Use `yolov8x.pt` for better accuracy
- Adjust `--confidence` threshold based on your needs
- Use `--frame-logic` for better detection validation

## Development Notes

### Key Components

1. **Object Detection**: YOLO models for initial detection
2. **Object Tracking**: ByteTrack for frame-to-frame tracking
3. **Line Crossing Logic**: Custom logic to detect line crossings
4. **Frame Validation**: Filters out unreliable detections
5. **Results Export**: Comprehensive CSV and video outputs

### Areas for Improvement

- Handbag detection accuracy (currently ~44%)
- Real-time processing optimization
- Multi-line support enhancement
- GUI interface development

## Support

For questions or issues:
1. Check the logs in the `logs/` directory
2. Review the analysis tools in `analysis_tools/`
3. Adjust parameters in `src/config.py`
4. Test with different model sizes and confidence thresholds

## License

This project is for internal use. Please respect any licensing requirements for the YOLO models and other dependencies. 