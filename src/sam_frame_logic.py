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
                 lines: List[sv.LineZone],
                 fps: float = 30.0,
                 line_x_positions: List[int] = None,
                 min_safe_time: float = 0.1,    # Much more relaxed
                 min_uncertain_time: float = 0.05, # Much more relaxed
                 min_very_brief_time: float = 0.01): # Much more relaxed
        """Initialize SAM segment tracker."""
        # Use provided line x-positions or extract from LineZone objects
        if line_x_positions is not None:
            self.line_x_positions = line_x_positions
        else:
            # Extract x-coordinates from LineZone objects
            self.line_x_positions = []
            
            for i, line_zone in enumerate(lines):
                # LineZone objects have start and end points, extract x-coordinate from start
                if hasattr(line_zone, 'start') and hasattr(line_zone.start, 'x'):
                    x_pos = line_zone.start.x
                    self.line_x_positions.append(x_pos)
                elif hasattr(line_zone, 'start') and hasattr(line_zone.start, 'x_coordinate'):
                    x_pos = line_zone.start.x_coordinate
                    self.line_x_positions.append(x_pos)
                elif hasattr(line_zone, 'start') and hasattr(line_zone.start, 'coordinates'):
                    # Try to access coordinates as a tuple
                    coords = line_zone.start.coordinates
                    if hasattr(coords, '__getitem__') and len(coords) > 0:
                        x_pos = coords[0]
                        self.line_x_positions.append(x_pos)
                    else:
                        self.line_x_positions.append(0)  # Default fallback
                else:
                    # Fallback: try to access as a point or tuple
                    if hasattr(line_zone, 'x'):
                        x_pos = line_zone.x
                        self.line_x_positions.append(x_pos)
                    elif isinstance(line_zone, (list, tuple)) and len(line_zone) > 0:
                        x_pos = line_zone[0]
                        self.line_x_positions.append(x_pos)
                    else:
                        self.line_x_positions.append(0)  # Default fallback
        
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
        
        # Track which crossings have already been detected to prevent duplicates
        self.detected_crossings = set()  # (track_id, line_id, direction)
        self.recent_crossings = []  # Store recent crossings with positions for spatial deduplication
        
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
        """Check if segment crossed any lines using mask centroids."""
        crossings = []
        
        if len(self.segment_history[track_id]) < 2:
            return crossings
        
        # Get current mask centroid
        current_centroid = segment_info['centroid']
        current_center_x = current_centroid[0]
        
        history = list(self.segment_history[track_id])
        
        # Check crossings with recent history
        for i in range(len(history) - 1):
            if i + 1 >= len(history):
                break
                
            # Get previous centroid
            prev_frame = history[i]
            curr_frame = history[i + 1]
            
            # For current detection, use segment_info centroid
            if i + 1 == len(history) - 1:
                curr_center_x = current_center_x
            else:
                curr_center_x = curr_frame['centroid'][0]
            
            # Get previous center
            prev_center_x = prev_frame['centroid'][0]
            
            # Check each line using extracted x-coordinates
            for line_idx, line_x in enumerate(self.line_x_positions):
                crossing_info = self._detect_line_crossing_simple(
                    prev_center_x, curr_center_x, line_x, track_id, 
                    segment_info['class'], line_idx + 1, segment_info['frame_idx']
                )
                
                if crossing_info:
                    print(f"✅ CROSSING: Track {track_id} ({segment_info['class']}) crossed Line {line_idx + 1} ({crossing_info['direction']})")
                    crossings.append(crossing_info)
                    self.line_crossings.append(crossing_info)
        
        return crossings
    
    def _detect_line_crossing_simple(self, prev_x, curr_x, line_x, track_id, obj_class, line_id, frame_idx):
        """Improved line crossing detection using x coordinates."""
        # Add tolerance for line crossing detection - MUCH MORE RELAXED
        tolerance = 50  # Reduced from 200 to 50 pixels for more accurate detection
        min_movement = 20  # Minimum movement distance to consider as crossing
        
        # Check if movement is significant enough
        movement_distance = abs(curr_x - prev_x)
        if movement_distance < min_movement:
            return None
        
        # Check if crossed vertical line with tolerance
        line_crossed = False
        crossing_type = "none"
        
        # Only detect direct crossing, not proximity
        if (prev_x < line_x < curr_x) or (curr_x < line_x < prev_x):
            line_crossed = True
            crossing_type = "direct_cross"
        
        if line_crossed:
            direction = "IN" if prev_x < curr_x else "OUT"
            
            # Check if this crossing has already been detected (stricter duplicate prevention)
            crossing_key = (track_id, line_id, direction)
            if crossing_key in self.detected_crossings:
                return None
            
            # Very strict time-based deduplication - only 1 object of same type per 3 frames
            time_threshold = 3  # Only 3 frames allowed between same object type crossings
            
            # Clean up old recent crossings (keep only very recent)
            self.recent_crossings = [
                c for c in self.recent_crossings 
                if abs(c['frame_idx'] - frame_idx) <= time_threshold
            ]
            
            # Check for same object type crossing within 2 frames
            for recent in self.recent_crossings:
                if recent['class'] == obj_class:
                    frame_diff = abs(recent['frame_idx'] - frame_idx)
                    print(f"🚫 Too frequent: {obj_class} crossing blocked (last crossing {frame_diff} frames ago, need 3+ frames gap)")
                    return None
            
            # Get track duration for validation
            track_duration = self._get_track_duration(track_id, frame_idx)
            
            # Add confidence based on movement distance
            confidence = min(1.0, movement_distance / 10.0)  # More reasonable confidence calculation
            
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
                'movement_distance': movement_distance,
                'crossing_type': crossing_type
            }
            
            # Mark this crossing as detected
            self.detected_crossings.add(crossing_key)
            
            # Add to recent crossings for spatial deduplication
            self.recent_crossings.append({
                'class': obj_class,
                'line_id': line_id,
                'curr_x': curr_x,
                'frame_idx': frame_idx
            })
            
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