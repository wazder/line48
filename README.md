# LineLogic - Object Tracking and Line Crossing Detection

A computer vision system for detecting and counting objects crossing virtual lines in video streams. The system uses YOLO models for object detection and ByteTrack for object tracking, with advanced frame-based validation logic.

## Project Structure

```
LineLogic/
├── src/                    # Main source code
│   ├── run_analysis.py     # Main analysis runner
│   ├── logic.py           # Basic tracking logic
│   ├── frame_logic.py     # Frame-based validation
│   ├── config.py          # Configuration settings
│   ├── utils.py           # Utility functions
│   └── video_config.py    # Video selection utilities
├── analysis_tools/         # Analysis and debugging scripts
│   ├── compare_to_ground_truth.py
│   ├── compare_results.py
│   ├── analyze_misses.py
│   └── ... (other analysis scripts)
├── models/                 # YOLO model files
├── videos/                 # Input video files
├── outputs/                # Processed video outputs
├── logs/                   # Detection logs
├── training/               # Training data and scripts
└── envs/                   # Python virtual environments
```

## Target Classes

- **person** - People crossing the lines
- **backpack** - Backpacks carried by people
- **handbag** - Handbags and purses
- **suitcase** - Luggage and suitcases

## Key Features

- **Command-line interface** - Flexible parameter configuration via command line
- **Automatic video discovery** - Finds videos in videos/, New Videos 2/, and cropped_videos/ directories
- **Interactive video selection** - Numbered menu for easy video selection
- **Video cropping with preview** - Interactive crop selection with visual markers
- **Configurable parameters** - All model and logic parameters adjustable via command line
- **Frame-based validation** - Filters out brief, unreliable detections
- **Multi-line detection** - Supports multiple virtual lines
- **Class-specific thresholds** - Different validation rules per object class
- **Comprehensive logging** - Detailed logs for analysis and debugging
- **Enhanced CSV output** - Comprehensive results with parameters and statistics
- **Visual overlays** - Real-time display of detections and counts
- **Performance evaluation** - Tools to compare against ground truth

## Quick Start

1. **Setup Environment:**
   ```bash
   # Activate virtual environment
   envs\lov10-env310\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **List Available Videos:**
   ```bash
   cd src
   python run_analysis.py --list-videos
   ```

3. **Run Analysis (Interactive):**
   ```bash
   python run_analysis.py
   ```

4. **Run Analysis (Command Line):**
   ```bash
   # Process specific video with frame logic
   python run_analysis.py --video "../videos/New Videos 2/IMG_0015_blurred.MOV" --frame-logic
   
   # Process with custom parameters
   python run_analysis.py --video "../videos/New Videos 2/IMG_0015_blurred.MOV" --confidence 0.3 --iou 0.4 --imgsz 1280
   ```

5. **Create Cropped Video:**
   ```bash
   python run_analysis.py --crop-video "../videos/New Videos 2/IMG_0015_blurred.MOV"
   ```

6. **Compare Results:**
   ```bash
   python analysis_tools/compare_to_ground_truth.py
   ```

## Performance

Current performance on test video:
- **Person:** 95% accuracy (84/84 detected)
- **Backpack:** 79% accuracy (30/38 detected)
- **Handbag:** 44% accuracy (24/54 detected) - Main challenge
- **Suitcase:** 85% accuracy (11/13 detected)

## Analysis Tools

The `analysis_tools/` directory contains scripts for:
- Comparing model results against ground truth
- Analyzing missed detections
- Debugging confidence distributions
- Performance optimization

## Configuration

### Command-Line Parameters

All parameters can be configured via command line:

**Model Parameters:**
- `--confidence`: YOLO confidence threshold (default: 0.25)
- `--iou`: NMS IoU threshold (default: 0.45)
- `--imgsz`: Input image size (default: 1024)

**Frame Logic Parameters:**
- `--min-safe-time`: Minimum time for safe predictions (default: 0.5s)
- `--min-uncertain-time`: Minimum time for uncertain predictions (default: 0.28s)
- `--min-very-brief-time`: Minimum time for very brief predictions (default: 0.17s)

**Logic Selection:**
- `--frame-logic`: Use frame-based validation (default)
- `--basic-logic`: Use basic tracking logic

### File Configuration

Additional parameters can be adjusted in `src/config.py`:
- Line positions and spacing
- Output paths and logging
- Video selection defaults

## Dependencies

- ultralytics (YOLO models)
- supervision (tracking and visualization)
- opencv-python (video processing)
- numpy (numerical operations)
- pandas (data analysis)
- argparse (command-line interface)
- subprocess (FFmpeg integration)

## Output Files

The system generates several output files:

- **Processed Video**: `outputs/[video_name]_processed_[timestamp].mp4`
- **Detection Log**: `logs/[video_name]_log_[timestamp].csv`
- **Results Summary**: `logs/[video_name]_results_[timestamp].csv`
- **Cropped Videos**: `videos/cropped_videos/[video_name]_cropped.mp4`

The results CSV includes comprehensive analysis data with parameters, detection results, success rates, and analysis notes. 