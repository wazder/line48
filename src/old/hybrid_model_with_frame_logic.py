import sys
import os

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

import supervision as sv
from config import SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, LOG_CSV_PATH, LINE_POINTS, LINE_HEIGHT, LINE_IDS
from supervision import VideoInfo
import csv
import numpy as np
import cv2
from utils import load_model
from frame_logic import FrameBasedTracker

# Load model
model_single = load_model()

# Load video info to get fps
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(f"📦 Total frames in video: {video_info.total_frames}")
print(f"Video FPS: {video_info.fps}")

# Initialize frame-based tracker
tracker = FrameBasedTracker(
    fps=video_info.fps,
    min_safe_time=0.5,      # seconds - safe prediction
    min_mixed_time=0.28,    # seconds - uncertain prediction  
    min_detection_time=0.18  # seconds - minimum to consider
)

# Setup lines and annotators
LINES = [sv.LineZone(start=p, end=sv.Point(p.x, LINE_HEIGHT)) for p in LINE_POINTS]
line_annotators = [
    sv.LineZoneAnnotator(
        display_in_count=False,
        display_out_count=False,
        text_thickness=2,
        text_scale=1.0
    )
    for _ in LINE_IDS
]

# Model setup
COCO_NAMES = model_single.model.names
SELECTED_CLASSES = ["person", "backpack", "handbag", "suitcase"]
SELECTED_CLASS_IDS = [k for k, v in COCO_NAMES.items() if v in SELECTED_CLASSES]

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    if index % 100 == 0:
        print(f"🟢 Processed frame {index}")

    # Run inference
    results = model_single(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter to selected classes
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    
    # Update tracker
    detections = tracker.byte_tracker.update_with_detections(detections)
    
    # Update object presence tracking
    tracker.update_object_presence(detections, index, COCO_NAMES)
    
    # Process line crossings with frame-based logic
    tracker.process_line_crossing(detections, index, LINES, LINE_IDS, COCO_NAMES)
    
    # Visual annotations
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # Get tracking duration for color coding
        tid = detections.tracker_id[i]
        if tid is not None:
            duration_frames, duration_seconds = tracker.get_presence_duration(tid, index)
            confidence = tracker.classify_prediction_confidence(duration_frames)
            color = tracker.get_confidence_color(confidence)
            
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(
                frame, f"{tid}", (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
    
    # Draw lines
    for line_idx, line in enumerate(LINES):
        line_annotators[line_idx].annotate(frame, line)
    
    return frame

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
print("\n📊 Per-class counts with confidence levels:")
results = tracker.get_results_summary()
for class_name, counts in results.items():
    safe = counts["safe"]
    uncertain = counts["uncertain"]
    total = counts["total"]
    print(f"{class_name:<10} → Safe: {safe}, Uncertain: {uncertain}, Total: {total}")

# Export enhanced CSV log
with open(LOG_CSV_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Object ID", "Class", "Line Number", "Direction", "Frame", 
        "Timestamp (min:sec)", "Confidence", "Tracking Duration"
    ])
    writer.writerows(tracker.get_log_rows())

print(f"\n📁 Enhanced log saved to {LOG_CSV_PATH}")
print(f"🎯 Frame-based logic implemented with {video_info.fps} FPS thresholds") 