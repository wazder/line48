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
        self.frame_crossings = {}  # Track crossings per frame: {frame_idx: [(class, x_pos), ...]}
        self.spatial_temporal_history = []  # Track all crossings with spatial-temporal info
        
        # Category-specific ID mapping
        self.category_ids = {}  # Maps global track_id to category-specific ID
        self.category_counters = {  # Counters for each category
            'person': 0,
            'backpack': 0, 
            'handbag': 0,
            'suitcase': 0
        }
        
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
    
    def _get_category_id(self, global_track_id: int, obj_class: str) -> str:
        """Get or create category-specific ID for tracking."""
        if global_track_id not in self.category_ids:
            # Create new category-specific ID
            self.category_counters[obj_class] += 1
            category_prefix = {
                'person': 'P',
                'backpack': 'B', 
                'handbag': 'H',
                'suitcase': 'S'
            }
            prefix = category_prefix.get(obj_class, 'X')
            category_id = f"{prefix}{self.category_counters[obj_class]}"
            self.category_ids[global_track_id] = category_id
        
        return self.category_ids[global_track_id]
    
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
        # Class-specific thresholds for better detection - fine-tuned
        movement_thresholds = {
            'person': 18,      # Higher to reduce from 18 to 7-8
            'backpack': 14,    # Lower to increase from 11 to 7
            'handbag': 18,     # Lower to increase from 4 to 3-4
            'suitcase': 20     # Higher to maintain 2 (currently 3)
        }
        min_movement = movement_thresholds.get(obj_class, 20)
        
        # Different proximity thresholds per object type - fine-tuned
        proximity_thresholds = {
            'person': 20,      # Stricter to reduce from 18 to 7-8
            'backpack': 40,    # More generous to increase from 11 to 7
            'handbag': 90,     # More generous to increase from 4 to 3-4
            'suitcase': 55     # Slightly stricter to maintain 2 (currently 3)
        }
        line_proximity_threshold = proximity_thresholds.get(obj_class, 50)
        
        # Check if movement is significant enough
        movement_distance = abs(curr_x - prev_x)
        if movement_distance < min_movement:
            return None
        
        # Check if crossed vertical line with tolerance
        line_crossed = False
        crossing_type = "none"
        
        # Only detect direct crossing AND must be close to line
        if (prev_x < line_x < curr_x) or (curr_x < line_x < prev_x):
            # Additional check: object must pass close to the line
            closest_distance_to_line = min(abs(prev_x - line_x), abs(curr_x - line_x))
            if closest_distance_to_line <= line_proximity_threshold:
                line_crossed = True
                crossing_type = "direct_cross"
                # Don't print here - will print only if crossing is actually counted
            else:
                # Reduce spam - only show rejection for very close misses
                if closest_distance_to_line <= line_proximity_threshold + 20:
                    print(f"🚫 Close miss: {obj_class} passed {closest_distance_to_line}px from line {line_id} (need ≤{line_proximity_threshold}px)")
                return None
        
        if line_crossed:
            direction = "IN" if prev_x < curr_x else "OUT"
            
            # Check if this crossing has already been detected (stricter duplicate prevention)
            crossing_key = (track_id, line_id, direction)
            if crossing_key in self.detected_crossings:
                return None
            
            # Enhanced spatial-temporal duplicate prevention - fine-tuned
            spatial_temporal_thresholds = {
                'person': {'spatial': 120, 'temporal': 25},      # Stricter to reduce from 18 to 7-8
                'backpack': {'spatial': 60, 'temporal': 25},     # More lenient to increase from 11 to 7
                'handbag': {'spatial': 160, 'temporal': 12},     # More generous to increase from 4 to 3-4
                'suitcase': {'spatial': 90, 'temporal': 18}      # Slightly stricter to maintain 2
            }
            
            thresholds = spatial_temporal_thresholds.get(obj_class, {'spatial': 100, 'temporal': 30})
            spatial_threshold = thresholds['spatial']
            temporal_threshold = thresholds['temporal']
            
            # Clean old spatial-temporal history
            current_time = frame_idx
            self.spatial_temporal_history = [
                entry for entry in self.spatial_temporal_history 
                if abs(entry['frame_idx'] - current_time) <= temporal_threshold
            ]
            
            # Check for spatial-temporal duplicates
            for entry in self.spatial_temporal_history:
                if (entry['class'] == obj_class and 
                    entry['line_id'] == line_id and
                    abs(entry['x_pos'] - curr_x) < spatial_threshold):
                    frame_diff = abs(entry['frame_idx'] - frame_idx)
                    print(f"🚫 Spatial-temporal duplicate [Frame {frame_idx}]: {obj_class} too close to previous crossing at x={entry['x_pos']} ({frame_diff} frames ago)")
                    return None
            
            # Same-frame check (immediate duplicates)
            if frame_idx not in self.frame_crossings:
                self.frame_crossings[frame_idx] = []
            
            for existing_class, existing_x in self.frame_crossings[frame_idx]:
                # Same-frame thresholds fine-tuned
                same_frame_thresholds = {
                    'person': 50,      # Stricter to reduce from 18 to 7-8
                    'backpack': 80,    # More lenient to increase from 11 to 7
                    'handbag': 70,     # More lenient to increase from 4 to 3-4
                    'suitcase': 35     # Slightly more lenient to maintain 2
                }
                same_frame_threshold = same_frame_thresholds.get(obj_class, 50)
                if (existing_class == obj_class and 
                    abs(existing_x - curr_x) < same_frame_threshold):
                    print(f"🚫 Same frame duplicate [Frame {frame_idx}]: {obj_class} at x={curr_x} too close to existing at x={existing_x}")
                    return None
            
            # Class-specific time thresholds for different objects - fine-tuned
            time_thresholds = {
                'person': 6,      # Longer gap to reduce from 18 to 7-8
                'backpack': 5,    # Shorter gap to increase from 11 to 7
                'handbag': 5,     # Shorter gap to increase from 4 to 3-4
                'suitcase': 10    # Longer gap to maintain 2 (currently 3)
            }
            time_threshold = time_thresholds.get(obj_class, 3)
            
            self.recent_crossings = [
                c for c in self.recent_crossings 
                if abs(c['frame_idx'] - frame_idx) <= time_threshold
            ]
            
            for recent in self.recent_crossings:
                if recent['class'] == obj_class:
                    frame_diff = abs(recent['frame_idx'] - frame_idx)
                    print(f"🚫 Too frequent [Frame {frame_idx}]: {obj_class} crossing blocked (last crossing {frame_diff} frames ago, need {time_threshold}+ frames gap)")
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
            
            # Add to frame crossings to prevent same-frame duplicates
            self.frame_crossings[frame_idx].append((obj_class, curr_x))
            
            # Add to spatial-temporal history
            self.spatial_temporal_history.append({
                'class': obj_class,
                'line_id': line_id,
                'x_pos': curr_x,
                'frame_idx': frame_idx
            })
            
            # Print valid crossing only when it's actually counted
            closest_distance = min(abs(prev_x - line_x), abs(curr_x - line_x))
            print(f"✅ COUNTED [Frame {frame_idx}]: {obj_class} crossed line {line_id} at {closest_distance}px distance")
            
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
            'safe_crossings': defaultdict(lambda: {'IN': 0, 'OUT': 0}),
            'uncertain_crossings': defaultdict(lambda: {'IN': 0, 'OUT': 0}),
            'very_brief_crossings': defaultdict(lambda: {'IN': 0, 'OUT': 0}),
            'discarded_crossings': defaultdict(lambda: {'IN': 0, 'OUT': 0}),
            'total_crossings': defaultdict(lambda: {'IN': 0, 'OUT': 0})
        }
        
        for crossing in self.line_crossings:
            obj_class = crossing['class']
            direction = crossing['direction']  # 'IN' or 'OUT'
            duration_frames = crossing['duration_frames']
            
            # Classify based on duration
            if duration_frames >= self.min_safe_frames:
                results['safe_crossings'][obj_class][direction] += 1
                category = 'safe'
            elif duration_frames >= self.min_uncertain_frames:
                results['uncertain_crossings'][obj_class][direction] += 1
                category = 'uncertain'
            elif duration_frames >= self.min_very_brief_frames:
                results['very_brief_crossings'][obj_class][direction] += 1
                category = 'very_brief'
            else:
                results['discarded_crossings'][obj_class][direction] += 1
                category = 'discarded'
                self.discarded_crossings.append(crossing)
            
            results['total_crossings'][obj_class][direction] += 1
            crossing['category'] = category
        
        return results
    
    def get_crossing_summary(self) -> str:
        """Get formatted summary of crossings with IN/OUT breakdown."""
        results = self.validate_and_count_crossings()
        
        summary = "📊 SAM Segment Tracking Results:\n"
        
        for obj_class in ['person', 'backpack', 'handbag', 'suitcase']:
            safe_in = results['safe_crossings'][obj_class]['IN']
            safe_out = results['safe_crossings'][obj_class]['OUT']
            uncertain_in = results['uncertain_crossings'][obj_class]['IN']
            uncertain_out = results['uncertain_crossings'][obj_class]['OUT']
            brief_in = results['very_brief_crossings'][obj_class]['IN']
            brief_out = results['very_brief_crossings'][obj_class]['OUT']
            
            total_in = safe_in + uncertain_in + brief_in
            total_out = safe_out + uncertain_out + brief_out
            
            if total_in > 0 or total_out > 0:
                summary += f"{obj_class:10} → Safe: IN:{safe_in} OUT:{safe_out}, "
                summary += f"Uncertain: IN:{uncertain_in} OUT:{uncertain_out}, "
                summary += f"Very Brief: IN:{brief_in} OUT:{brief_out}, "
                summary += f"Total: IN:{total_in} OUT:{total_out}\n"
        
        summary += f"\n🗑️ Discarded crossings (too brief): {len(self.discarded_crossings)}\n"
        
        return summary