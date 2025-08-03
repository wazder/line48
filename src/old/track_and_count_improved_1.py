import supervision as sv
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import cv2
import os
import csv
from datetime import timedelta


# Video paths
SOURCE_VIDEO_PATH = r"C:\Users\murat\Desktop\StajHW\LineLogic\MVI_6817_blurred.MOV"
TARGET_VIDEO_PATH = "hybrid_model_testing_1.mp4"
LOG_CSV_PATH = "logs/hybrid_model_testing_1.csv"

# Defining the lines to cover door entrance/frame. 
BASE_X = 960
LINE_SPACING = 125
LINE_HEIGHT = 1080


LINE_POINTS = [
    sv.Point(BASE_X - 2*(LINE_SPACING), 0), # Line 1
    sv.Point(BASE_X - LINE_SPACING, 0),     # Line 2
    sv.Point(BASE_X, 0),                    # Line 3
    sv.Point(BASE_X + LINE_SPACING, 0),     # Line 4
    sv.Point(BASE_X + (2*LINE_SPACING), 0)  # Line 5
]
LINES = [sv.LineZone(start=p, end=sv.Point(p.x, LINE_HEIGHT)) for p in LINE_POINTS]
LINE_IDS = [1, 2, 3, 4, 5]
line_annotators = [
    sv.LineZoneAnnotator(
        display_in_count=False,
        display_out_count=False,
        text_thickness=2,
        text_scale=1.0
    )
    for _ in LINE_IDS
]


# Load models
model_general = YOLO("yolo11x.pt").to("cuda")
model_suitcase = YOLO("suitcase.pt").to("cuda")
print("Using hbyrid model")
print("🧠 General model using:", model_general.device)
print("🧳 Suitcase model using:", model_suitcase.device)


def draw_text_with_background(img, text, org, font, scale, text_color, bg_color, thickness=1, alpha=0.6):
    """Draw text with a background rectangle"""
    # Get size of text
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    box_coords = ((x, y - text_height - 10), (x + text_width + 10, y + 10))

    # Create overlay for transparency
    overlay = img.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw text on top
    cv2.putText(img, text, (x + 5, y), font, scale, text_color, thickness, cv2.LINE_AA)

# Select classes 
COCO_NAMES = model_general.model.names
SUITCASE_COCO_ID = 28  # YOLO COCO class index for "suitcase"
SELECTED_CLASSES = ["person", "backpack", "handbag"]
SELECTED_CLASS_IDS = [k for k, v in COCO_NAMES.items() if v in SELECTED_CLASSES]

# Counters and memory
per_class_counter = defaultdict(lambda: {"in": 0, "out": 0})
counted_ids_in = set()
counted_ids_out = set()

# Logging
recent_message = ""
log_rows = []
os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)

# Tracker
byte_tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60)

# Video info
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(f"📦 Total frames in video: {video_info.total_frames}")

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Main logic
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global recent_message
    # if index >= 2000:
    #     raise StopIteration    

    if index % 100 == 0:
        print(f"🟢 Processed frame {index}")

    # Run both models
    results_general = model_general(frame, verbose=False)[0]
    results_suitcase = model_suitcase(frame, verbose=False)[0]
    det_general = sv.Detections.from_ultralytics(results_general)
    det_suitcase = sv.Detections.from_ultralytics(results_suitcase)
    print("Suitcase detections:", len(det_suitcase))
    print("Suitcase confidences:", det_suitcase.confidence.tolist())

    # Filter general detections to only selected classes
    det_general = det_general[np.isin(det_general.class_id, SELECTED_CLASS_IDS)]

    # Map suitcase class_id (0 in fine-tuned model) → 28 (YOLO COCO's suitcase ID)
    for i in range(len(det_suitcase)):
        det_suitcase.class_id[i] = SUITCASE_COCO_ID

    # Combine both sets
    detections = sv.Detections.merge([det_general, det_suitcase])
    detections = byte_tracker.update_with_detections(detections)

    # --- Remainder of your callback remains unchanged ---
    # Annotate detections and tracker IDs
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{detections.tracker_id[i]}", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Line crossing logic
    for line_idx, line in enumerate(LINES):
        crossed_in, crossed_out = line.trigger(detections)
        for i, is_in in enumerate(crossed_in):
            if is_in:
                tid = detections.tracker_id[i]
                cls = COCO_NAMES[detections.class_id[i]]
                if tid not in counted_ids_in:
                    counted_ids_in.add(tid)
                    per_class_counter[cls]["in"] += 1
                    timestamp = str(timedelta(seconds=int(index / video_info.fps)))
                    msg = f'Object with ID "{tid}" crossed line "{LINE_IDS[line_idx]}" and detected as "{cls}" at time {timestamp}, at frame {index}'
                    print(msg)
                    recent_message = msg
                    log_rows.append([tid, cls, LINE_IDS[line_idx], "IN", index, timestamp])

        for i, is_out in enumerate(crossed_out):
            if is_out:
                tid = detections.tracker_id[i]
                cls = COCO_NAMES[detections.class_id[i]]
                if tid not in counted_ids_out:
                    counted_ids_out.add(tid)
                    per_class_counter[cls]["out"] += 1
                    timestamp = str(timedelta(seconds=int(index / video_info.fps)))
                    msg = f'Object with ID "{tid}" crossed line "{LINE_IDS[line_idx]}" and detected as "{cls}" at time {timestamp}, at frame {index}'
                    print(msg)
                    recent_message = msg
                    log_rows.append([tid, cls, LINE_IDS[line_idx], "OUT", index, timestamp])

    # Draw annotations
    frame = trace_annotator.annotate(scene=frame, detections=detections)
    frame = box_annotator.annotate(scene=frame, detections=detections)
    labels = [
        f"#{tid} {COCO_NAMES[cls]} {conf:.2f}"
        for conf, cls, tid in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    for annotator, line in zip(line_annotators, LINES):
        frame = annotator.annotate(frame, line)

    for line_id, line in zip(LINE_IDS, LINES):
        pos = (line.vector.start.x - 20, 40)
        cv2.putText(
            frame,
            f"{line_id}",
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

    # Total in/out
    cv2.putText(frame, f"In: {len(counted_ids_in)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Out: {len(counted_ids_out)}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Per-class counter
    y0 = 150
    for i, (class_name, counts) in enumerate(per_class_counter.items()):
        cv2.putText(frame, f"{class_name} IN: {counts['in']}", (50, y0 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} OUT: {counts['out']}", (250, y0 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Bottom left: latest crossing message
    if recent_message:
        draw_text_with_background(
            frame,
            recent_message,
            org=(30, frame.shape[0] - 30),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            scale=0.7,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
            thickness=2
        )

    # Top right: frame number
    frame_text = f"Frame: {index}"
    draw_text_with_background(
        frame,
        frame_text,
        org=(frame.shape[1] - 200, 40),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale=0.9,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        thickness=2
    )

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

# Final report
print("Processing complete.")
print("\n📊 Per-class counts:")
for class_name, counts in per_class_counter.items():
    print(f"{class_name:<10} → IN: {counts['in']}, OUT: {counts['out']}")

# Export CSV
with open(LOG_CSV_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Class", "Line Number", "Direction", "Frame", "Timestamp (min:sec)"])
    writer.writerows(log_rows)

print(f"\n📁 Log saved to {LOG_CSV_PATH}")
