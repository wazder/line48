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
    
    def __init__(self, fps, min_safe_time=0.3, min_uncertain_time=0.15, min_very_brief_time=0.08):
        """
        Initialize frame-based tracker.
        Args:
            fps: Video frame rate
            min_safe_time: Minimum time (seconds) for safe prediction (reduced from 0.5 to 0.3)
            min_uncertain_time: Minimum time (seconds) for uncertain prediction (reduced from 0.28 to 0.15)
            min_very_brief_time: Minimum time (seconds) for very brief prediction (reduced from 0.167 to 0.08)
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
        # Frame thresholds initialized - logging removed for cleaner output
    
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
    
    def classify_prediction_confidence(self, duration_frames, class_name=None):
        """Classify prediction confidence based on duration, with special handling for persons."""
        # Special handling for persons - they should be counted even with shorter durations
        if class_name == "person":
            # More lenient thresholds for persons
            if duration_frames >= self.min_safe_frames // 2:  # Half the normal safe time
                return "safe"
            elif duration_frames >= self.min_uncertain_frames // 2:  # Half the normal uncertain time
                return "uncertain"
            elif duration_frames >= self.min_very_brief_frames // 2:  # Half the normal very brief time
                return "very_brief"
            else:
                return "discard"
        else:
            # Standard thresholds for other objects
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
        
        # Special debug for persons
        person_detections = []
        for i in range(len(detections)):
            if i < len(detections.class_id):
                cls = class_names[detections.class_id[i]]
                if cls == "person":
                    tid = detections.tracker_id[i] if i < len(detections.tracker_id) else None
                    person_detections.append((i, tid))
        
        if person_detections:
            print(f"👥 Frame {current_frame}: Found {len(person_detections)} person(s)")
            for idx, tid in person_detections:
                print(f"   - Person ID: {tid}")
        
        for line_idx, line in enumerate(lines):
            crossed_in, crossed_out = line.trigger(detections)
            
            # Debug: Print all detections for this frame
            if len(detections) > 0:
                print(f"🔍 Frame {current_frame}: {len(detections)} detections")
                for i in range(len(detections)):
                    if i < len(detections.class_id):
                        cls = class_names[detections.class_id[i]]
                        tid = detections.tracker_id[i] if i < len(detections.tracker_id) else None
                        print(f"   - {cls} (ID: {tid})")
            
            # IN crossings
            for i, is_in in enumerate(crossed_in):
                if is_in:
                    tid = detections.tracker_id[i]
                    cls = class_names[detections.class_id[i]]
                    
                    # Special handling for persons - always log their crossing attempts
                    if cls == "person":
                        print(f"🎯 PERSON CROSSING ATTEMPT: Person (ID: {tid}) attempting to cross line {line_ids[line_idx]} IN")
                    
                    if tid is not None and tid not in self.global_in:
                        self.global_in.add(tid)
                        duration_frames, duration_seconds = self.get_presence_duration(tid, current_frame)
                        confidence = self.classify_prediction_confidence(duration_frames, cls)
                        timestamp = str(timedelta(seconds=int(current_frame / self.fps)))
                        
                        # Debug: Print crossing detection
                        print(f"✅ IN CROSSING: {cls} (ID: {tid}) crossed line {line_ids[line_idx]} - Duration: {duration_seconds:.2f}s - Confidence: {confidence}")
                        
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
                            print(f"❌ DISCARDED: {cls} (ID: {tid}) - Duration too short ({duration_seconds:.2f}s)")
                            # For persons, still count them even if discarded
                            if cls == "person":
                                print(f"🎯 SPECIAL: Counting person anyway due to importance")
                                self.per_class_counter[cls]["very_brief"] += 1
                                self.per_class_counter[cls]["total"] += 1
                                self.log_rows.append([
                                    tid, cls, line_ids[line_idx], "IN", current_frame, 
                                    timestamp, "very_brief", f"{duration_seconds:.2f}s"
                                ])
                            else:
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
                    cls = class_names[detections.class_id[i]]
                    
                    # Special handling for persons - always log their crossing attempts
                    if cls == "person":
                        print(f"🎯 PERSON CROSSING ATTEMPT: Person (ID: {tid}) attempting to cross line {line_ids[line_idx]} OUT")
                    
                    if tid is not None and tid not in self.global_out:
                        self.global_out.add(tid)
                        duration_frames, duration_seconds = self.get_presence_duration(tid, current_frame)
                        confidence = self.classify_prediction_confidence(duration_frames, cls)
                        timestamp = str(timedelta(seconds=int(current_frame / self.fps)))
                        
                        # Debug: Print crossing detection
                        print(f"✅ OUT CROSSING: {cls} (ID: {tid}) crossed line {line_ids[line_idx]} - Duration: {duration_seconds:.2f}s - Confidence: {confidence}")
                        
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
                            print(f"❌ DISCARDED: {cls} (ID: {tid}) - Duration too short ({duration_seconds:.2f}s)")
                            # For persons, still count them even if discarded
                            if cls == "person":
                                print(f"🎯 SPECIAL: Counting person anyway due to importance")
                                self.per_class_counter[cls]["very_brief"] += 1
                                self.per_class_counter[cls]["total"] += 1
                                self.log_rows.append([
                                    tid, cls, line_ids[line_idx], "OUT", current_frame, 
                                    timestamp, "very_brief", f"{duration_seconds:.2f}s"
                                ])
                            else:
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