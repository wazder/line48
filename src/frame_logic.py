import sys
import os

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

import supervision as sv
import numpy as np
from collections import defaultdict
from datetime import timedelta
import cv2


class FrameBasedTracker:
    """
    Enhanced tracker with frame-based validation logic.
    Tracks object presence duration and classifies predictions based on tracking time.
    """
    
    def __init__(self, fps, min_safe_time=0.5, min_uncertain_time=0.28, min_very_brief_time=0.167):
        """
        Initialize frame-based tracker.
        Args:
            fps: Video frame rate
            min_safe_time: Minimum time (seconds) for safe prediction
            min_uncertain_time: Minimum time (seconds) for uncertain prediction
            min_very_brief_time: Minimum time (seconds) for very brief prediction
        """
        self.fps = fps
        self.min_safe_frames = int(min_safe_time * fps)      # e.g., 27
        self.min_uncertain_frames = int(min_uncertain_time * fps)  # e.g., 15
        self.min_very_brief_frames = int(min_very_brief_time * fps)  # e.g., 9
        self.object_presence = {}  # {tracker_id: {"first_seen": frame, "last_seen": frame, "class": class_name}}
        self.counted_ids_in = set()
        self.counted_ids_out = set()
        self.global_in = set()  # Track first in per object
        self.global_out = set() # Track first out per object
        self.per_class_counter = defaultdict(lambda: {"safe": 0, "uncertain": 0, "very_brief": 0, "total": 0})
        self.log_rows = []
        self.byte_tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60)
        self.discarded_crossings = []  # List of (tid, cls, line_id, direction, frame, duration_frames)
        print(f"🔢 Frame thresholds for {fps} FPS:")
        print(f"   Safe (≥{self.min_safe_frames} frames)")
        print(f"   Uncertain ({self.min_uncertain_frames}-{self.min_safe_frames-1} frames)")
        print(f"   Very brief ({self.min_very_brief_frames}-{self.min_uncertain_frames-1} frames)")
        print(f"   Discard (<{self.min_very_brief_frames} frames)")
        print(f"   Time thresholds: Safe≥{min_safe_time}s, Uncertain≥{min_uncertain_time}s, Very brief≥{min_very_brief_time}s")
    
    def update_object_presence(self, detections, current_frame, class_names):
        current_ids = set()
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None:
                current_ids.add(tracker_id)
                class_name = class_names[detections.class_id[i]]
                if tracker_id not in self.object_presence:
                    self.object_presence[tracker_id] = {
                        "first_seen": current_frame,
                        "last_seen": current_frame,
                        "class": class_name
                    }
                else:
                    self.object_presence[tracker_id]["last_seen"] = current_frame
        to_remove = [tid for tid in self.object_presence.keys() if tid not in current_ids]
        for tid in to_remove:
            del self.object_presence[tid]
    
    def get_presence_duration(self, tracker_id, current_frame):
        if tracker_id not in self.object_presence:
            return 0, 0
        obj_info = self.object_presence[tracker_id]
        duration_frames = current_frame - obj_info["first_seen"]
        duration_seconds = duration_frames / self.fps
        return duration_frames, duration_seconds
    
    def classify_prediction_confidence(self, duration_frames):
        if duration_frames >= self.min_safe_frames:
            return "safe"
        elif duration_frames >= self.min_uncertain_frames:
            return "uncertain"
        elif duration_frames >= self.min_very_brief_frames:
            return "very_brief"
        else:
            return "discard"
    
    def get_confidence_color(self, confidence):
        if confidence == "safe":
            return (0, 255, 0)  # Green
        elif confidence == "uncertain":
            return (0, 255, 255)  # Yellow
        elif confidence == "very_brief":
            return (0, 128, 255)  # Orange
        else:
            return (128, 128, 128)  # Gray
    
    def process_line_crossing(self, detections, current_frame, lines, line_ids, class_names):
        crossings = []
        for line_idx, line in enumerate(lines):
            crossed_in, crossed_out = line.trigger(detections)
            # IN crossings
            for i, is_in in enumerate(crossed_in):
                if is_in:
                    tid = detections.tracker_id[i]
                    if tid is not None and tid not in self.global_in:
                        self.global_in.add(tid)
                        duration_frames, duration_seconds = self.get_presence_duration(tid, current_frame)
                        confidence = self.classify_prediction_confidence(duration_frames)
                        cls = class_names[detections.class_id[i]]
                        timestamp = str(timedelta(seconds=int(current_frame / self.fps)))
                        
                        # Add to crossings list for visualization
                        if len(detections.xyxy) > i:
                            bbox = detections.xyxy[i]
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            crossings.append({
                                'position': (center_x, center_y),
                                'class': cls,
                                'direction': 'IN',
                                'confidence': confidence,
                                'track_id': tid
                            })
                        
                        if confidence == "discard":
                            self.discarded_crossings.append((tid, cls, line_ids[line_idx], "IN", current_frame, duration_frames))
                        else:
                            if confidence == "safe":
                                self.counted_ids_in.add(tid)
                            self.per_class_counter[cls][confidence] += 1
                            self.per_class_counter[cls]["total"] += 1
                            self.log_rows.append([
                                tid, cls, line_ids[line_idx], "IN", current_frame, 
                                timestamp, confidence, f"{duration_seconds:.2f}s"
                            ])
            # OUT crossings
            for i, is_out in enumerate(crossed_out):
                if is_out:
                    tid = detections.tracker_id[i]
                    if tid is not None and tid not in self.global_out:
                        self.global_out.add(tid)
                        duration_frames, duration_seconds = self.get_presence_duration(tid, current_frame)
                        confidence = self.classify_prediction_confidence(duration_frames)
                        cls = class_names[detections.class_id[i]]
                        timestamp = str(timedelta(seconds=int(current_frame / self.fps)))
                        
                        # Add to crossings list for visualization
                        if len(detections.xyxy) > i:
                            bbox = detections.xyxy[i]
                            center_x = int((bbox[0] + bbox[2]) / 2)
                            center_y = int((bbox[1] + bbox[3]) / 2)
                            crossings.append({
                                'position': (center_x, center_y),
                                'class': cls,
                                'direction': 'OUT',
                                'confidence': confidence,
                                'track_id': tid
                            })
                        
                        if confidence == "discard":
                            self.discarded_crossings.append((tid, cls, line_ids[line_idx], "OUT", current_frame, duration_frames))
                        else:
                            if confidence == "safe":
                                self.counted_ids_out.add(tid)
                            self.per_class_counter[cls][confidence] += 1
                            self.per_class_counter[cls]["total"] += 1
                            self.log_rows.append([
                                tid, cls, line_ids[line_idx], "OUT", current_frame, 
                                timestamp, confidence, f"{duration_seconds:.2f}s"
                            ])
        
        return crossings
    def get_results_summary(self):
        return dict(self.per_class_counter)
    def get_log_rows(self):
        return self.log_rows.copy()
    def get_discarded_summary(self):
        return self.discarded_crossings.copy()
    
    def get_line_in_count(self, line_id):
        """Get count of objects that crossed the line IN direction."""
        count = 0
        for tid, cls, lid, direction, frame, duration in self.log_rows:
            if lid == line_id and direction == "IN":
                count += 1
        return count
    
    def get_line_out_count(self, line_id):
        """Get count of objects that crossed the line OUT direction."""
        count = 0
        for tid, cls, lid, direction, frame, duration in self.log_rows:
            if lid == line_id and direction == "OUT":
                count += 1
        return count 