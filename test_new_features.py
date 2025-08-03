#!/usr/bin/env python3
"""
Test script to demonstrate the new LineLogic command-line features.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"📝 Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="src")
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
        else:
            print("❌ Failed!")
            if result.stderr:
                print("📤 Error:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Test the new command-line features."""
    print("🚀 LineLogic Enhanced Features Test")
    print("=" * 60)
    
    # Test 1: Help
    run_command("python run_analysis.py --help", "Help command")
    
    # Test 2: List videos
    run_command("python run_analysis.py --list-videos", "List all available videos")
    
    # Test 3: Show parameter examples
    print(f"\n{'='*60}")
    print("📋 Available Command Examples:")
    print(f"{'='*60}")
    
    examples = [
        "# Interactive mode (select video and parameters)",
        "python run_analysis.py",
        "",
        "# Process specific video with frame logic",
        "python run_analysis.py --video \"../videos/New Videos 2/IMG_0015_blurred.MOV\" --frame-logic",
        "",
        "# Process with custom parameters",
        "python run_analysis.py --video \"../videos/New Videos 2/IMG_0015_blurred.MOV\" --confidence 0.3 --iou 0.4 --imgsz 1280",
        "",
        "# Use basic logic instead of frame logic",
        "python run_analysis.py --video \"../videos/New Videos 2/IMG_0015_blurred.MOV\" --basic-logic",
        "",
        "# Create cropped video",
        "python run_analysis.py --crop-video \"../videos/New Videos 2/IMG_0015_blurred.MOV\"",
        "",
        "# Process with custom frame logic parameters",
        "python run_analysis.py --video \"../videos/New Videos 2/IMG_0015_blurred.MOV\" --min-safe-time 0.6 --min-uncertain-time 0.3 --min-very-brief-time 0.2"
    ]
    
    for example in examples:
        print(example)
    
    print(f"\n{'='*60}")
    print("🎯 New Features Summary:")
    print(f"{'='*60}")
    print("✅ 1. Automatic video discovery in videos/, New Videos 2/, and cropped_videos/")
    print("✅ 2. Interactive video selection with numbered list")
    print("✅ 3. Video cropping with FFmpeg integration")
    print("✅ 4. Configurable model parameters (confidence, iou, imgsz)")
    print("✅ 5. Configurable frame logic parameters")
    print("✅ 6. Comprehensive CSV output with parameters and results")
    print("✅ 7. Command-line interface with argparse")
    print("✅ 8. Default parameter values matching current settings")
    
    print(f"\n{'='*60}")
    print("📁 Output Files:")
    print(f"{'='*60}")
    print("• Processed video: outputs/[video_name]_processed_[timestamp].mp4")
    print("• Detection log: logs/[video_name]_log_[timestamp].csv")
    print("• Results summary: logs/[video_name]_results_[timestamp].csv")
    print("• Cropped videos: videos/cropped_videos/[video_name]_cropped.mp4")

if __name__ == "__main__":
    main() 