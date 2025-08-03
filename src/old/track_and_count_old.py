import supervision as sv
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import cv2
import os

# Video paths
SOURCE_VIDEO_PATH = r"C:\Users\murat\Desktop\StajHW\LineLogic\MVI_6817_blurred.MOV"
TARGET_VIDEO_PATH = "logger_testing_1.mp4"

# Defining the lines to cover door entrance/frame. 
BASE_X = 960
LINE_SPACING = 125
LINE_HEIGHT = 1080

LINE_POINTS = [
    sv.Point(BASE_X - 2*(LINE_SPACING), 0), #LeftMost line
    sv.Point(BASE_X - LINE_SPACING, 0),  # Left line
    sv.Point(BASE_X, 0),                 # Center line
    sv.Point(BASE_X + LINE_SPACING, 0),   # Right line
    sv.Point(BASE_X + (2*LINE_SPACING), 0), # RightMost line
]
LINES = [sv.LineZone(start=p, end=sv.Point(p.x, LINE_HEIGHT)) for p in LINE_POINTS]
line_annotators = [sv.LineZoneAnnotator() for _ in LINES]

# Load model
model = YOLO("yolo11x.pt")
model.to("cuda")
print("YOLO is using:", model.device)

# Select classes 
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASSES = ['person', 'backpack', 'handbag', 'suitcase'] # Selected from Yolo Supervision's existing object classes.
SELECTED_CLASS_IDS = [k for k, v in CLASS_NAMES_DICT.items() if v in SELECTED_CLASSES]

# Counters and memory
per_class_counter = defaultdict(lambda: {"in": 0, "out": 0})
counted_ids_in = set()
counted_ids_out = set()

# Tracker, can be modified further for more accuracy.
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.2,
    lost_track_buffer=60
)
# For debugging and extra info
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(f"📦 Total frames in video: {video_info.total_frames}")

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

#Main logic
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # For early stopping/debugging.
    if index >= 200:
        raise StopIteration
    if index % 100 == 0: # Again, extra info
        print(f"🟢 Processed frame {index}")


    #init.
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    # Draw tracker points
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{detections.tracker_id[i]}", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Collect IDs that crossed IN or OUT on any line(main line logic)
    crossed_in_ids = set()
    crossed_out_ids = set()
    for line in LINES:
        crossed_in, crossed_out = line.trigger(detections)
        for i, is_in in enumerate(crossed_in):
            if is_in:
                crossed_in_ids.add(detections.tracker_id[i])
        for i, is_out in enumerate(crossed_out):
            if is_out:
                crossed_out_ids.add(detections.tracker_id[i])

    # Count entries/exits only once per object
    for i, track_id in enumerate(detections.tracker_id):
        class_id = detections.class_id[i]
        class_name = CLASS_NAMES_DICT[class_id]

        if track_id in crossed_in_ids and track_id not in counted_ids_in:
            per_class_counter[class_name]["in"] += 1
            counted_ids_in.add(track_id)
            print(f"Counted IN for {class_name} (id: {track_id})")

        if track_id in crossed_out_ids and track_id not in counted_ids_out:
            per_class_counter[class_name]["out"] += 1
            counted_ids_out.add(track_id)
            print(f"Counted OUT for {class_name} (id: {track_id})")

    # Annotations
    frame = trace_annotator.annotate(scene=frame, detections=detections)
    frame = box_annotator.annotate(scene=frame, detections=detections)
    labels = [
        f"#{tid} {CLASS_NAMES_DICT[cls]} {conf:.2f}"
        for conf, cls, tid in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    for annotator, line in zip(line_annotators, LINES):
        frame = annotator.annotate(frame, line)

    # Overlay counts
    cv2.putText(frame, f"In: {len(counted_ids_in)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Out: {len(counted_ids_out)}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Per-class counts
    y0 = 150
    for i, (class_name, counts) in enumerate(per_class_counter.items()):
        cv2.putText(frame, f"{class_name} IN: {counts['in']}", (50, y0 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} OUT: {counts['out']}", (250, y0 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame
#for early stopping/debugging
try:
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
except StopIteration:
    print("\n🛑 Stopped manually.")

# Final report
print("Processing complete.")
print("\n📊 Per-class counts:")
for class_name, counts in per_class_counter.items():
    print(f"{class_name:<10} → IN: {counts['in']}, OUT: {counts['out']}") 



2/2

