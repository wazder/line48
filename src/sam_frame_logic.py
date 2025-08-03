"""
SAM Frame Logic - Segment tracking and line crossing detection
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import supervision as sv
from collections import defaultdict, deque

class SAMSegmentTracker:
    """Tracks SAM segmentation masks across frames and detects line crossings."""
    
    def __init__(self, 
                 lines: List[sv.Point],
                 fps: float = 30.0,
                 min_safe_time: float = 0.5,    # Yumuşatıldı
                 min_uncertain_time: float = 0.3, # Yumuşatıldı
                 min_very_brief_time: float = 0.1): # Yumuşatıldı
        """Initialize SAM segment tracker."""
        self.lines = lines
        self.fps = fps
        
        # Time thresholds
        self.min_safe_time = min_safe_time
        self.min_uncertain_time = min_uncertain_time
        self.min_very_brief_time = min_very_brief_time
        
        # Convert time to frame thresholds
        self.min_safe_frames = int(min_safe_time * fps)
        self.min_uncertain_frames = int(min_uncertain_time * fps)
        self.min_very_brief_frames = int(min_very_brief_time * fps)
        
        # Tracking data structures
        self.active_segments = {}
        self.segment_history = defaultdict(deque)
        self.line_crossings = []
        self.discarded_crossings = []
        
        # Statistics
        self.stats = {
            'safe_crossings': defaultdict(int),
            'uncertain_crossings': defaultdict(int),
            'discarded_crossings': defaultdict(int),
            'total_segments': defaultdict(int)
        }
    
    def update(self, frame_idx: int, detections: List[Dict], tracked_objects=None) -> List[Dict]:
        """Update tracker with new detections and segmentation masks."""
        current_crossings = []
        
        # Update active segments
        for detection in detections:
            if 'mask' not in detection or detection['mask'] is None:
                continue
                
            # Use provided track ID from ByteTrack
            track_id = detection.get('track_id', -1)
            if track_id == -1:
                continue  # Skip untracked detections
            
            # Get segment centroid
            centroid_x, centroid_y = self._get_mask_centroid(detection['mask'])
            if centroid_x is None:
                continue
            
            segment_info = {
                'track_id': track_id,
                'class': detection['class'],
                'confidence': detection['confidence'],
                'mask': detection['mask'],
                'centroid': (centroid_x, centroid_y),
                'bbox': detection['bbox'],
                'frame_idx': frame_idx,
                'mask_area': np.sum(detection['mask'])
            }
            
            # Update segment history
            self.segment_history[track_id].append({
                'frame_idx': frame_idx,
                'centroid': (centroid_x, centroid_y),
                'bbox': detection['bbox'],
                'mask': detection['mask'],
                'class': detection['class']
            })
            
            # Keep only recent history
            max_history = max(30, self.min_safe_frames)
            if len(self.segment_history[track_id]) > max_history:
                self.segment_history[track_id].popleft()
            
            # Check line crossings
            crossings = self._check_line_crossings(track_id, segment_info)
            current_crossings.extend(crossings)
            
            # Update active segments
            self.active_segments[track_id] = segment_info
        
        # Clean up old segments
        self._cleanup_old_segments(frame_idx)
        
        return current_crossings
    
    def _get_mask_centroid(self, mask: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Get centroid of segmentation mask."""
        if mask is None:
            return None, None
            
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0:
            return None, None
            
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        return centroid_x, centroid_y
    
    def _check_line_crossings(self, track_id: int, segment_info: Dict) -> List[Dict]:
        """Check if segment crossed any lines using bbox centers."""
        crossings = []
        
        if len(self.segment_history[track_id]) < 2:
            return crossings
        
        # Get current and previous bbox centers
        current_bbox = segment_info['bbox']
        current_center_x = (current_bbox[0] + current_bbox[2]) / 2
        
        history = list(self.segment_history[track_id])
        
        # Check crossings with recent history
        for i in range(len(history) - 1):
            if i + 1 >= len(history):
                break
                
            # Get previous bbox center
            prev_frame = history[i]
            curr_frame = history[i + 1]
            
            # For current detection, use segment_info bbox
            if i + 1 == len(history) - 1:
                curr_center_x = current_center_x
            else:
                # Use stored bbox if available, otherwise use centroid
                if 'bbox' in curr_frame:
                    curr_bbox = curr_frame['bbox']
                    curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
                else:
                    curr_center_x = curr_frame['centroid'][0]
            
            # Get previous center
            if 'bbox' in prev_frame:
                prev_bbox = prev_frame['bbox']
                prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
            else:
                prev_center_x = prev_frame['centroid'][0]
            
            # Check each line using actual line points
            for line_idx, line_point in enumerate(self.lines):
                line_x = line_point.x
                crossing_info = self._detect_line_crossing_simple(
                    prev_center_x, curr_center_x, line_x, track_id, 
                    segment_info['class'], line_idx + 1, segment_info['frame_idx']
                )
                
                if crossing_info:
                    crossings.append(crossing_info)
                    self.line_crossings.append(crossing_info)
        
        return crossings
    
    def _detect_line_crossing_simple(self, prev_x, curr_x, line_x, track_id, obj_class, line_id, frame_idx):
        """Improved line crossing detection using x coordinates."""
        # Add tolerance for line crossing detection
        tolerance = 10  # pixels
        
        # Check if crossed vertical line with tolerance
        line_crossed = False
        if (prev_x < line_x < curr_x) or (curr_x < line_x < prev_x):
            line_crossed = True
        elif abs(prev_x - line_x) <= tolerance and abs(curr_x - line_x) <= tolerance:
            # Object is very close to line
            line_crossed = True
        
        if line_crossed:
            direction = "IN" if prev_x < curr_x else "OUT"
            
            # Get track duration for validation
            track_duration = self._get_track_duration(track_id, frame_idx)
            
            # Add confidence based on movement distance
            movement_distance = abs(curr_x - prev_x)
            confidence = min(1.0, movement_distance / 50.0)  # Normalize confidence
            
            crossing_info = {
                'track_id': track_id,
                'class': obj_class,
                'line_id': line_id,
                'direction': direction,
                'frame_idx': frame_idx,
                'duration_frames': track_duration,
                'duration_seconds': track_duration / self.fps,
                'position': (curr_x, 0),
                'prev_x': prev_x,
                'curr_x': curr_x,
                'line_x': line_x,
                'confidence': confidence,
                'movement_distance': movement_distance
            }
            
            print(f"🎯 Line crossing detected: {obj_class} (ID:{track_id}) crossed line {line_id} at frame {frame_idx}")
            print(f"   Movement: {prev_x:.1f} → {curr_x:.1f} (line at {line_x})")
            print(f"   Direction: {direction}, Confidence: {confidence:.2f}")
            
            return crossing_info
        
        return None
    
    def _get_track_duration(self, track_id: int, current_frame: int) -> int:
        """Get duration of track in frames."""
        if track_id not in self.segment_history:
            return 0
        
        history = list(self.segment_history[track_id])
        if not history:
            return 0
        
        first_frame = history[0]['frame_idx']
        return current_frame - first_frame + 1
    
    def _cleanup_old_segments(self, current_frame: int):
        """Remove old inactive segments."""
        inactive_tracks = []
        
        for track_id, segment in self.active_segments.items():
            if current_frame - segment['frame_idx'] > 30:  # 1 second at 30fps
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.active_segments[track_id]
    
    def validate_and_count_crossings(self) -> Dict:
        """Validate all crossings using frame-based logic and return counts."""
        results = {
            'safe_crossings': defaultdict(int),
            'uncertain_crossings': defaultdict(int), 
            'very_brief_crossings': defaultdict(int),
            'discarded_crossings': defaultdict(int),
            'total_crossings': defaultdict(int)
        }
        
        for crossing in self.line_crossings:
            obj_class = crossing['class']
            duration_frames = crossing['duration_frames']
            
            # Classify based on duration
            if duration_frames >= self.min_safe_frames:
                results['safe_crossings'][obj_class] += 1
                category = 'safe'
            elif duration_frames >= self.min_uncertain_frames:
                results['uncertain_crossings'][obj_class] += 1
                category = 'uncertain'
            elif duration_frames >= self.min_very_brief_frames:
                results['very_brief_crossings'][obj_class] += 1
                category = 'very_brief'
            else:
                results['discarded_crossings'][obj_class] += 1
                category = 'discarded'
                self.discarded_crossings.append(crossing)
            
            results['total_crossings'][obj_class] += 1
            crossing['category'] = category
        
        return results
    
    def get_crossing_summary(self) -> str:
        """Get formatted summary of crossings."""
        results = self.validate_and_count_crossings()
        
        summary = "📊 SAM Segment Tracking Results:\n"
        
        for obj_class in ['person', 'backpack', 'handbag', 'suitcase']:
            safe = results['safe_crossings'][obj_class]
            uncertain = results['uncertain_crossings'][obj_class]
            very_brief = results['very_brief_crossings'][obj_class]
            discarded = results['discarded_crossings'][obj_class]
            total = safe + uncertain + very_brief
            
            if total > 0:
                summary += f"{obj_class:10} → Safe: {safe}, Uncertain: {uncertain}, Very Brief: {very_brief}, Total: {total}\n"
        
        summary += f"\n🗑️ Discarded crossings (too brief): {len(self.discarded_crossings)}\n"
        
        return summary