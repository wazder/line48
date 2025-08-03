"""
Example usage of LineLogic with both basic and frame-based tracking logic.
This script demonstrates how to use the modular approach.
"""

import sys
import os

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

import supervision as sv
from config import SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, LOG_CSV_PATH
from supervision import VideoInfo
import csv
from utils import load_model
from logic import create_callback, create_callback_with_frame_logic

def run_basic_tracking():
    """Run tracking with basic logic (original approach)"""
    print("🔧 Running with basic tracking logic...")
    
    # Load model and video info
    model_single = load_model()
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    
    # Create callback with basic logic
    callback = create_callback(model_single, video_info)
    
    # Process video
    output_path = TARGET_VIDEO_PATH.replace('.mp4', '_basic.mp4')
    try:
        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=output_path,
            callback=callback
        )
    except StopIteration:
        print("\n🛑 Stopped manually.")
    
    # Print results
    print("\n📊 Basic tracking results:")
    for class_name, counts in callback.per_class_counter.items():
        print(f"{class_name:<10} → IN: {counts['in']}, OUT: {counts['out']}")
    
    # Export CSV log
    log_path = LOG_CSV_PATH.replace('.csv', '_basic.csv')
    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Class", "Line Number", "Direction", "Frame", "Timestamp (min:sec)"])
        writer.writerows(callback.log_rows)
    
    print(f"📁 Basic log saved to {log_path}")
    return output_path, log_path

def run_frame_based_tracking():
    """Run tracking with frame-based logic (enhanced approach)"""
    print("🎯 Running with frame-based tracking logic...")
    
    # Load model and video info
    model_single = load_model()
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    
    # Create callback with frame-based logic
    callback = create_callback_with_frame_logic(model_single, video_info, use_frame_logic=True)
    
    # Process video
    output_path = TARGET_VIDEO_PATH.replace('.mp4', '_frame_logic.mp4')
    try:
        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=output_path,
            callback=callback
        )
    except StopIteration:
        print("\n🛑 Stopped manually.")
    
    # Print results
    print("\n📊 Frame-based tracking results:")
    results = callback.per_class_counter()
    for class_name, counts in results.items():
        safe = counts["safe"]
        uncertain = counts["uncertain"]
        total = counts["total"]
        print(f"{class_name:<10} → Safe: {safe}, Uncertain: {uncertain}, Total: {total}")
    
    # Export enhanced CSV log
    log_path = LOG_CSV_PATH.replace('.csv', '_frame_logic.csv')
    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Object ID", "Class", "Line Number", "Direction", "Frame", 
            "Timestamp (min:sec)", "Confidence", "Tracking Duration"
        ])
        writer.writerows(callback.log_rows())
    
    print(f"📁 Frame-based log saved to {log_path}")
    return output_path, log_path

if __name__ == "__main__":
    print("🚀 LineLogic Example Usage")
    print("=" * 50)
    
    # Choose which approach to run
    choice = input("Choose approach:\n1. Basic tracking\n2. Frame-based tracking\n3. Both (for comparison)\nEnter choice (1/2/3): ")
    
    if choice == "1":
        run_basic_tracking()
    elif choice == "2":
        run_frame_based_tracking()
    elif choice == "3":
        print("\n🔄 Running both approaches for comparison...")
        basic_output, basic_log = run_basic_tracking()
        print("\n" + "="*50)
        frame_output, frame_log = run_frame_based_tracking()
        
        print(f"\n📊 Comparison Summary:")
        print(f"Basic tracking: {basic_output}")
        print(f"Frame-based tracking: {frame_output}")
        print(f"Compare logs: {basic_log} vs {frame_log}")
    else:
        print("Invalid choice. Running frame-based tracking by default...")
        run_frame_based_tracking() 