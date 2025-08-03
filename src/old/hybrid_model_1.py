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
from logic import create_callback
from utils import load_model


# Load models
model_single = load_model()# Load video info to get fps
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(f"📦 Total frames in video: {video_info.total_frames}")
print(f"Video FPS: {video_info.fps}")

# Create callback with models and video info
callback = create_callback(model_single, video_info)

# Process video
try:
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
except StopIteration:
    print("\n🛑 Stopped manually.")

# Print results
print("\n📊 Per-class counts:")
for class_name, counts in callback.per_class_counter.items():
    print(f"{class_name:<10} → IN: {counts['in']}, OUT: {counts['out']}")

# Export CSV log
with open(LOG_CSV_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Class", "Line Number", "Direction", "Frame", "Timestamp (min:sec)"])
    writer.writerows(callback.log_rows)

print(f"\n📁 Log saved to {LOG_CSV_PATH}")
