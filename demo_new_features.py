#!/usr/bin/env python3
"""
LineLogic Enhanced Features Demonstration
Shows how to use the new command-line interface and features.
"""

import os
import sys

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_command(cmd, description):
    """Print a command with description."""
    print(f"\n📝 {description}:")
    print(f"   {cmd}")

def main():
    """Demonstrate the new LineLogic features."""
    print("🚀 LineLogic Enhanced Features Demonstration")
    print("=" * 60)
    
    print_section("1. Video Discovery and Listing")
    print("The system now automatically discovers videos in multiple directories:")
    print("   • videos/ (main directory)")
    print("   • videos/New Videos 2/ (subdirectory)")
    print("   • videos/cropped_videos/ (for cropped videos)")
    
    print_command(
        "python src/run_analysis.py --list-videos",
        "List all available videos with file sizes"
    )
    
    print_section("2. Interactive Video Selection")
    print("When no video is specified, the system shows an interactive menu:")
    print("   • Numbered list of all videos")
    print("   • File sizes displayed")
    print("   • Easy selection by number")
    
    print_command(
        "python src/run_analysis.py",
        "Interactive mode - select video and parameters"
    )
    
    print_section("3. Video Cropping with Preview")
    print("New feature: Create cropped videos with visual preview:")
    print("   • Shows video with center point marked")
    print("   • Displays 200-pixel markers from center")
    print("   • Interactive crop area selection")
    print("   • Uses FFmpeg for high-quality cropping")
    
    print_command(
        "python src/run_analysis.py --crop-video \"videos/New Videos 2/IMG_0015_blurred.MOV\"",
        "Create cropped video with interactive preview"
    )
    
    print_section("4. Configurable Model Parameters")
    print("All model parameters can now be adjusted via command line:")
    
    print_command(
        "python src/run_analysis.py --video \"videos/New Videos 2/IMG_0015_blurred.MOV\" --confidence 0.3 --iou 0.4 --imgsz 1280",
        "Process with custom confidence, IoU, and image size"
    )
    
    print("Available model parameters:")
    print("   • --confidence: YOLO confidence threshold (default: 0.25)")
    print("   • --iou: NMS IoU threshold (default: 0.45)")
    print("   • --imgsz: Input image size (default: 1024)")
    
    print_section("5. Frame Logic Parameters")
    print("Frame-based validation parameters can be customized:")
    
    print_command(
        "python src/run_analysis.py --video \"videos/New Videos 2/IMG_0015_blurred.MOV\" --min-safe-time 0.6 --min-uncertain-time 0.3 --min-very-brief-time 0.2",
        "Process with custom frame logic thresholds"
    )
    
    print("Available frame logic parameters:")
    print("   • --min-safe-time: Minimum time for safe predictions (default: 0.5s)")
    print("   • --min-uncertain-time: Minimum time for uncertain predictions (default: 0.28s)")
    print("   • --min-very-brief-time: Minimum time for very brief predictions (default: 0.17s)")
    
    print_section("6. Logic Selection")
    print("Choose between frame-based and basic logic:")
    
    print_command(
        "python src/run_analysis.py --video \"videos/New Videos 2/IMG_0015_blurred.MOV\" --frame-logic",
        "Use frame-based logic (recommended, default)"
    )
    
    print_command(
        "python src/run_analysis.py --video \"videos/New Videos 2/IMG_0015_blurred.MOV\" --basic-logic",
        "Use basic tracking logic"
    )
    
    print_section("7. Enhanced Output Files")
    print("The system now generates comprehensive output files:")
    print("   • Processed video: outputs/[video_name]_processed_[timestamp].mp4")
    print("   • Detection log: logs/[video_name]_log_[timestamp].csv")
    print("   • Results summary: logs/[video_name]_results_[timestamp].csv")
    print("   • Cropped videos: videos/cropped_videos/[video_name]_cropped.mp4")
    
    print("\nThe results CSV now includes:")
    print("   • All analysis parameters with descriptions")
    print("   • Detection results with success rates")
    print("   • Summary statistics")
    print("   • Analysis notes and recommendations")
    
    print_section("8. Complete Example Workflow")
    print("Here's a complete workflow example:")
    
    workflow_steps = [
        "1. List available videos:",
        "   python src/run_analysis.py --list-videos",
        "",
        "2. Create a cropped video:",
        "   python src/run_analysis.py --crop-video \"videos/New Videos 2/IMG_0015_blurred.MOV\"",
        "",
        "3. Process the cropped video with custom parameters:",
        "   python src/run_analysis.py --video \"videos/cropped_videos/IMG_0015_blurred_cropped.mp4\" --confidence 0.3 --iou 0.4 --imgsz 1280 --min-safe-time 0.6",
        "",
        "4. Check the results in the logs directory"
    ]
    
    for step in workflow_steps:
        print(step)
    
    print_section("9. Help and Documentation")
    print_command(
        "python src/run_analysis.py --help",
        "Show complete help with examples"
    )
    
    print("\n🎉 All features are now ready to use!")
    print("The system is much more flexible and user-friendly than before.")

if __name__ == "__main__":
    main() 