"""
SAM Frame Logic - Segment tracking and line crossing detection with frame-based validation
Extends the original frame_logic.py to work with SAM segmentation masks.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import supervision as sv
from collections import defaultdict, deque
import time

class SAMSegmentTracker:
    """
    Tracks SAM segmentation masks across frames and detects line crossings.
    Uses frame-based validation to filter out brief, unreliable detections.
    """
    
    def __init__(self, 
                 lines: List[sv.Point],
                 fps: float = 30.0,
                 min_safe_time: float = 0.5,
                 min_uncertain_time: float = 0.28,
                 min_very_brief_time: float = 0.17):
        """
        Initialize SAM segment tracker.
        
        Args:
            lines: List of line points for crossing detection
            fps: Video FPS for time calculations
            min_safe_time: Minimum time for safe predictions (seconds)
            min_uncertain_time: Minimum time for uncertain predictions (seconds)
            min_very_brief_time: Minimum time for very brief predictions (seconds)
        """
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
        self.active_segments = {}  # track_id -> segment_info
        self.segment_history = defaultdict(deque)  # track_id -> deque of positions
        self.line_crossings = []
        self.discarded_crossings = []
        
        # Statistics
        self.stats = {
            'safe_crossings': defaultdict(int),
            'uncertain_crossings': defaultdict(int),
            'discarded_crossings': defaultdict(int),
            'total_segments': defaultdict(int)
        }
        
        print(f"🎯 SAM Segment Tracker initialized")
        print(f"   FPS: {fps}")
        print(f"   Frame thresholds:")
        print(f"     Safe: ≥{self.min_safe_frames} frames (≥{min_safe_time}s)")
        print(f"     Uncertain: {self.min_uncertain_frames}-{self.min_safe_frames-1} frames ({min_uncertain_time}-{min_safe_time}s)")
        print(f"     Very brief: {self.min_very_brief_frames}-{self.min_uncertain_frames-1} frames ({min_very_brief_time}-{min_uncertain_time}s)")
        print(f"     Discard: <{self.min_very_brief_frames} frames (<{min_very_brief_time}s)")
    
    def update(self, frame_idx: int, detections: List[Dict], tracked_objects=None) -> List[Dict]:
        """
        Update tracker with new detections and segmentation masks.
        
        Args:
            frame_idx: Current frame index
            detections: List of detection dictionaries with masks
            tracked_objects: Supervision tracked objects (optional)
            
        Returns:
            List of valid crossings detected in this frame
        """
        current_crossings = []
        
        # Update active segments
        for detection in detections:
            if 'mask' not in detection or detection['mask'] is None:
                continue
                
            # Generate track ID (simplified - in practice use proper tracking)
            track_id = self._generate_track_id(detection, frame_idx)
            
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
                'mask': detection['mask'],
                'class': detection['class']
            })
            
            # Keep only recent history (for memory efficiency)
            max_history = max(100, self.min_safe_frames * 2)
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
    
    def _generate_track_id(self, detection: Dict, frame_idx: int) -> int:
        """
        Generate track ID for detection (simplified approach).
        In practice, this should use proper object tracking like ByteTrack.
        """
        # Simplified: use centroid + class as ID
        centroid_x, centroid_y = self._get_mask_centroid(detection['mask'])
        
        # Find closest existing track
        min_distance = float('inf')
        closest_track = None
        
        for track_id, segment in self.active_segments.items():
            if segment['class'] == detection['class']:
                prev_x, prev_y = segment['centroid']
                distance = np.sqrt((centroid_x - prev_x)**2 + (centroid_y - prev_y)**2)
                
                if distance < min_distance and distance < 100:  # Max distance threshold
                    min_distance = distance
                    closest_track = track_id
        
        if closest_track is not None:
            return closest_track
        else:
            # Create new track ID
            return max(self.active_segments.keys(), default=0) + 1
    
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
        """Check if segment crossed any lines."""
        crossings = []
        
        if len(self.segment_history[track_id]) < 2:
            return crossings
        
        # Get current and previous positions
        current_pos = segment_info['centroid']
        history = list(self.segment_history[track_id])
        
        # Check crossings with recent history
        for i in range(len(history) - 1):
            prev_pos = history[i]['centroid']
            curr_pos = history[i + 1]['centroid']
            
            # Check each line
            for line_idx, line_points in enumerate(self._get_line_pairs()):
                crossing_info = self._detect_line_crossing(
                    prev_pos, curr_pos, line_points, track_id, 
                    segment_info['class'], line_idx + 1, segment_info['frame_idx']
                )
                
                if crossing_info:
                    crossings.append(crossing_info)
        
        return crossings
    
    def _get_line_pairs(self) -> List[Tuple[sv.Point, sv.Point]]:
        """Convert line points to pairs for crossing detection."""
        line_pairs = []
        for i in range(0, len(self.lines), 2):
            if i + 1 < len(self.lines):
                # Vertical lines
                line_pairs.append((
                    sv.Point(self.lines[i].x, 0),
                    sv.Point(self.lines[i].x, 1080)  # Assume 1080p height
                ))
        return line_pairs
    
    def _detect_line_crossing(self, prev_pos, curr_pos, line_points, track_id, obj_class, line_id, frame_idx):
        """Detect if movement crosses a line."""
        x1, y1 = prev_pos
        x2, y2 = curr_pos
        
        # Line coordinates
        line_x = line_points[0].x
        
        # Check if crossed vertical line
        if (x1 < line_x < x2) or (x2 < line_x < x1):
            direction = "IN" if x1 < x2 else "OUT"
            
            # Get track duration for validation
            track_duration = self._get_track_duration(track_id, frame_idx)
            
            crossing_info = {
                'track_id': track_id,
                'class': obj_class,
                'line_id': line_id,
                'direction': direction,
                'frame_idx': frame_idx,
                'duration_frames': track_duration,
                'duration_seconds': track_duration / self.fps,
                'position': curr_pos
            }
            
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
        """
        Validate all crossings using frame-based logic and return counts.
        """
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