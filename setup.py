#!/usr/bin/env python3
"""
Setup script for LineLogic project
This script helps set up the project environment and download required models.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'models',
        'videos',
        'outputs',
        'logs',
        'videos/cropped_videos'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_model(url, filename):
    """Download a model file from URL."""
    models_dir = Path('models')
    model_path = models_dir / filename
    
    if model_path.exists():
        print(f"✓ Model already exists: {filename}")
        return True
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def download_models():
    """Download required YOLO models."""
    models = {
        'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
        'yolov11n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt',
        'yolov11x.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11x.pt'
    }
    
    print("\nDownloading YOLO models...")
    for filename, url in models.items():
        download_model(url, filename)

def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        'ultralytics',
        'supervision',
        'opencv-python',
        'numpy',
        'torch',
        'torchvision'
    ]
    
    print("\nChecking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def create_sample_config():
    """Create a sample configuration file."""
    config_content = '''# Sample configuration for LineLogic
# Copy this to your own config file and modify as needed

# Line configuration
LINES = [
    {
        'name': 'Entry Line',
        'start': (100, 200),
        'end': (500, 200),
        'direction': 'up'
    },
    {
        'name': 'Exit Line', 
        'start': (100, 400),
        'end': (500, 400),
        'direction': 'down'
    }
]

# Model settings
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMG_SIZE = 1024

# Frame logic settings
MIN_SAFE_TIME = 0.5
MIN_UNCERTAIN_TIME = 0.28
MIN_VERY_BRIEF_TIME = 0.17
'''
    
    config_file = Path('sample_config.py')
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write(config_content)
        print("✓ Created sample_config.py")

def main():
    """Main setup function."""
    print("LineLogic Setup Script")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Download models
    download_models()
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing.")
        print("Run: pip install -r requirements.txt")
    
    print("\nNext steps:")
    print("1. Add your video files to the 'videos/' directory")
    print("2. Run: cd src && python run_analysis.py")
    print("3. Check SHARING_README.md for detailed usage instructions")

if __name__ == "__main__":
    main() 