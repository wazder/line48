import sys
import os
import numpy as np
import cv2
from datetime import timedelta
from collections import defaultdict
import supervision as sv
from config import LOG_CSV_PATH, LINE_POINTS, LINE_HEIGHT, LINE_IDS
from utils import draw_text_with_background


# Setup lines and annotators globally (constants)
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

# Create output directory for logs if missing
os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)


def create_callback(model_single, video_info, MAX_FRAMES=None, line_config=None):
    """
    Factory that creates and returns the callback function
    with access to the single YOLO model and video info.
    """

    COCO_NAMES = model_single.model.names 
    SELECTED_CLASSES = ["person", "backpack", "handbag", "suitcase"]
    SELECTED_CLASS_IDS = [k for k, v in COCO_NAMES.items() if v in SELECTED_CLASSES]

    # Setup lines based on configuration
    if line_config:
        # Use custom line configuration
        line_positions = line_config['line_positions']
        line_height = line_config['line_height']
        custom_lines = [sv.LineZone(start=sv.Point(x, 0), end=sv.Point(x, line_height)) for x in line_positions]
        custom_line_ids = list(range(1, len(line_positions) + 1))
        custom_line_annotators = [
            sv.LineZoneAnnotator(
                display_in_count=False,
                display_out_count=False,
                text_thickness=2,
                text_scale=1.0
            )
            for _ in custom_line_ids
        ]
    else:
        # Use default lines from config
        from config import LINE_POINTS, LINE_HEIGHT, LINE_IDS
        custom_lines = [sv.LineZone(start=p, end=sv.Point(p.x, LINE_HEIGHT)) for p in LINE_POINTS]
        custom_line_ids = LINE_IDS
        custom_line_annotators = line_annotators

    # Tracking and counting state
    byte_tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60)
    per_class_counter = defaultdict(lambda: {"in": 0, "out": 0})
    counted_ids_in = set()
    counted_ids_out = set()
    global_in = set()  # Track first in per object
    global_out = set() # Track first out per object
    recent_message = ""
    log_rows = []

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    # Use the MAX_FRAMES parameter passed to the function
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        nonlocal recent_message  # modify outer variable
        if MAX_FRAMES is not None and index >= MAX_FRAMES:
            raise StopIteration

        if index % 100 == 0:
            print(f"🟢 Processed frame {index}")

        # Run inference on frame using single model
        results = model_single(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections to only selected classes
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

        # Enhanced overlap suppression: exclude person objects, suppress overlaps across all classes
        from utils import is_box_inside
        if len(detections) > 1:
            keep = np.ones(len(detections), dtype=bool)
            for i in range(len(detections)):
                for j in range(len(detections)):
                    if i == j or not keep[i] or not keep[j]:
                        continue
                    
                    # Get class names for both objects
                    class_i = COCO_NAMES[detections.class_id[i]]
                    class_j = COCO_NAMES[detections.class_id[j]]
                    
                    # Skip if either object is a person (person objects are always preserved)
                    if class_i == "person" or class_j == "person":
                        continue
                    
                    # Check for overlap between any two objects (same or different classes)
                    box_i = detections.xyxy[i]
                    box_j = detections.xyxy[j]
                    conf_i = detections.confidence[i]
                    conf_j = detections.confidence[j]
                    
                    # If objects overlap >80%, keep the one with higher confidence
                    if is_box_inside(box_i, box_j, threshold=0.8):
                        if conf_i < conf_j:
                            keep[i] = False
                        else:
                            keep[j] = False
            detections = detections[keep]

        # Update tracker with filtered detections
        detections = byte_tracker.update_with_detections(detections)

        # Annotate tracked detections with tracker IDs
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(
                frame, f"{detections.tracker_id[i]}", (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Line crossing detection
        for line_idx, line in enumerate(custom_lines):
            crossed_in, crossed_out = line.trigger(detections)

            for i, is_in in enumerate(crossed_in):
                if is_in:
                    tid = detections.tracker_id[i]
                    if tid not in global_in:
                        global_in.add(tid)
                        cls = COCO_NAMES[detections.class_id[i]]
                        counted_ids_in.add(tid)
                        per_class_counter[cls]["in"] += 1
                        timestamp = str(timedelta(seconds=int(index / video_info.fps)))
                        msg = (
                            f'Object with ID "{tid}" crossed line "{custom_line_ids[line_idx]}" '
                            f'and detected as "{cls}" at time {timestamp}, at frame {index}'
                        )
                        print(msg)
                        recent_message = msg
                        log_rows.append([tid, cls, custom_line_ids[line_idx], "IN", index, timestamp])

            for i, is_out in enumerate(crossed_out):
                if is_out:
                    tid = detections.tracker_id[i]
                    if tid not in global_out:
                        global_out.add(tid)
                        cls = COCO_NAMES[detections.class_id[i]]
                        counted_ids_out.add(tid)
                        per_class_counter[cls]["out"] += 1
                        timestamp = str(timedelta(seconds=int(index / video_info.fps)))
                        msg = (
                            f'Object with ID "{tid}" crossed line "{custom_line_ids[line_idx]}" '
                            f'and detected as "{cls}" at time {timestamp}, at frame {index}'
                        )
                        print(msg)
                        recent_message = msg
                        log_rows.append([tid, cls, custom_line_ids[line_idx], "OUT", index, timestamp])

        # Draw annotations on frame
        frame = trace_annotator.annotate(scene=frame, detections=detections)
        frame = box_annotator.annotate(scene=frame, detections=detections)

        labels = [
            f"#{tid} {COCO_NAMES[cls]} {conf:.2f}"
            for conf, cls, tid in zip(detections.confidence, detections.class_id, detections.tracker_id)
            if tid is not None
        ]
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        for annotator, line in zip(custom_line_annotators, custom_lines):
            frame = annotator.annotate(frame, line)

        for line_id, line in zip(custom_line_ids, custom_lines):
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

        # Total in/out counters on frame
        cv2.putText(
            frame, f"In: {len(counted_ids_in)}", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )
        cv2.putText(
            frame, f"Out: {len(counted_ids_out)}", (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

        # Per-class in/out counts
        y0 = 150
        for i, (class_name, counts) in enumerate(per_class_counter.items()):
            cv2.putText(
                frame, f"{class_name} IN: {counts['in']}", (50, y0 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"{class_name} OUT: {counts['out']}", (250, y0 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        # Bottom left: last crossing message
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

    # Attach some attributes to callback to allow main.py to access these after processing:
    callback.per_class_counter = per_class_counter
    callback.log_rows = log_rows

    return callback


def create_callback_with_frame_logic(model_single, video_info, use_frame_logic=False):
    """
    Factory that creates and returns the callback function with optional frame-based logic.
    
    Args:
        model_single: YOLO model instance
        video_info: Video information object
        use_frame_logic: Whether to use frame-based validation logic
    """
    
    if use_frame_logic:
        # Import frame logic module
        try:
            from frame_logic import FrameBasedTracker
            print("🎯 Using frame-based logic for enhanced tracking")
            
            # Initialize frame-based tracker
            tracker = FrameBasedTracker(
                fps=video_info.fps,
                min_safe_time=0.5,
                min_uncertain_time=0.28,
                min_very_brief_time=0.167
            )
            
            # Setup lines and annotators
            from config import LINE_POINTS, LINE_HEIGHT, LINE_IDS
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
                crossings = tracker.process_line_crossing(detections, index, LINES, LINE_IDS, COCO_NAMES)
                
                # Visual annotations with confidence colors
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
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

            # Attach state to callback for external access
            callback.per_class_counter = tracker.get_results_summary
            callback.log_rows = tracker.get_log_rows
            callback.recent_message = lambda: "Frame-based logic active"
            
            return callback
            
        except ImportError:
            print("⚠️ Frame logic module not found, falling back to basic logic")
            return create_callback(model_single, video_info)
    else:
        # Use basic logic
        return create_callback(model_single, video_info)
