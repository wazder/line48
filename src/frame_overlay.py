"""
Frame Overlay System for Line Detection Display
Shows detected line crossings and object information at the bottom of frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import supervision as sv

class FrameOverlay:
    """Adds information overlay to frames showing line detection results."""
    
    def __init__(self, frame_height: int, frame_width: int, overlay_height: int = 150):
        """
        Initialize frame overlay system.
        
        Args:
            frame_height: Original frame height
            frame_width: Original frame width
            overlay_height: Height of the overlay area at bottom
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.overlay_height = overlay_height
        self.total_height = frame_height + overlay_height
        
        # Colors for different elements
        self.colors = {
            'background': (40, 40, 40),      # Dark gray
            'text': (255, 255, 255),         # White
            'highlight': (0, 255, 255),      # Yellow
            'success': (0, 255, 0),          # Green
            'warning': (0, 165, 255),        # Orange
            'error': (0, 0, 255),            # Red
            'person': (255, 150, 100),       # Light blue (BGR format)
            'backpack': (0, 255, 0),         # Green
            'handbag': (255, 0, 255),        # Magenta
            'suitcase': (0, 255, 255)        # Yellow
        }
        
        # Font settings - IMPROVED for better readability
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.small_font_scale = 0.6
        self.small_font_thickness = 2
        self.tiny_font_scale = 0.5
        self.tiny_font_thickness = 1
        
        # Frame Overlay initialized - logging removed for cleaner output
    
    def create_overlay_frame(self, original_frame: np.ndarray) -> np.ndarray:
        """
        Create frame with overlay area at bottom.
        
        Args:
            original_frame: Original frame
            
        Returns:
            Frame with overlay area
        """
        # Create new frame with overlay area
        overlay_frame = np.zeros((self.total_height, self.frame_width, 3), dtype=np.uint8)
        
        # Copy original frame to top
        overlay_frame[:self.frame_height, :] = original_frame
        
        # Fill overlay area with background color
        overlay_frame[self.frame_height:, :] = self.colors['background']
        
        return overlay_frame
    
    def add_line_info(self, frame: np.ndarray, line_crossings: List[Dict], 
                      current_frame: int, total_frames: int, sam_tracker=None) -> np.ndarray:
        """
        Add line crossing information to overlay.
        
        Args:
            frame: Frame with overlay area
            line_crossings: List of line crossing detections
            current_frame: Current frame number
            total_frames: Total number of frames
            
        Returns:
            Frame with line information added
        """
        # Add frame progress
        progress_text = f"Frame: {current_frame}/{total_frames}"
        cv2.putText(frame, progress_text, (10, self.frame_height + 25), 
                   self.font, self.small_font_scale, self.colors['text'], 
                   self.small_font_thickness)
        
        # Add line crossing information
        if line_crossings:
            # Group crossings by line
            line_groups = {}
            for crossing in line_crossings:
                line_id = crossing.get('line_id', 0)
                if line_id not in line_groups:
                    line_groups[line_id] = []
                line_groups[line_id].append(crossing)
            
            # Display line information
            y_offset = 50
            for line_id in sorted(line_groups.keys()):
                crossings = line_groups[line_id]
                
                # Line header
                line_text = f"Line {line_id}:"
                cv2.putText(frame, line_text, (10, self.frame_height + y_offset), 
                           self.font, self.small_font_scale, self.colors['highlight'], 
                           self.small_font_thickness)
                
                # Object information for this line
                obj_y_offset = y_offset + 20
                for i, crossing in enumerate(crossings[:3]):  # Show max 3 objects per line
                    obj_class = crossing.get('class', 'unknown')
                    direction = crossing.get('direction', 'unknown')
                    track_id = crossing.get('track_id', 'N/A')
                    confidence = crossing.get('confidence', 0.0)
                    
                    # Color based on class
                    color = self.colors.get(obj_class, self.colors['text'])
                    
                    # Get category ID if available
                    category_id = crossing.get('category_id', track_id)
                    if sam_tracker and hasattr(sam_tracker, '_get_category_id') and track_id != 'N/A':
                        try:
                            category_id = sam_tracker._get_category_id(int(track_id), obj_class)
                        except:
                            category_id = track_id
                    
                    # Object info text - handle confidence as string or float
                    try:
                        conf_value = float(confidence) if confidence is not None else 0.0
                        obj_text = f"  {obj_class} ({direction}) - ID:{category_id} - Conf:{conf_value:.2f}"
                    except (ValueError, TypeError):
                        obj_text = f"  {obj_class} ({direction}) - ID:{category_id} - Conf:{confidence}"
                    
                    cv2.putText(frame, obj_text, (20, self.frame_height + obj_y_offset), 
                               self.font, self.small_font_scale, color, 
                               self.small_font_thickness)
                    
                    obj_y_offset += 15
                
                y_offset = obj_y_offset + 10
                
                # If more objects, show count
                if len(crossings) > 3:
                    more_text = f"  ... and {len(crossings) - 3} more"
                    cv2.putText(frame, more_text, (20, self.frame_height + y_offset), 
                               self.font, self.small_font_scale, self.colors['warning'], 
                               self.small_font_thickness)
                    y_offset += 20
        else:
            # No crossings detected
            no_crossings_text = "No line crossings detected in this frame"
            cv2.putText(frame, no_crossings_text, (10, self.frame_height + 50), 
                       self.font, self.small_font_scale, self.colors['text'], 
                       self.small_font_thickness)
        
        return frame
    
    def add_detection_summary(self, frame: np.ndarray, detections: List[Dict], sam_tracker=None) -> np.ndarray:
        """
        Add detection summary to overlay.
        
        Args:
            frame: Frame with overlay area
            detections: List of detections
            
        Returns:
            Frame with detection summary added
        """
        # Organize detections by class with category IDs
        class_detections = {}
        for detection in detections:
            obj_class = detection.get('class', 'unknown')
            track_id = detection.get('track_id', 'N/A')
            
            # Get category-specific ID
            category_id = detection.get('category_id', track_id)
            if sam_tracker and hasattr(sam_tracker, '_get_category_id') and track_id != 'N/A':
                try:
                    category_id = sam_tracker._get_category_id(int(track_id), obj_class)
                except:
                    category_id = track_id
            
            if obj_class not in class_detections:
                class_detections[obj_class] = []
            class_detections[obj_class].append(category_id)
        
        # Display detection summary with track IDs
        summary_text = "Detections:"
        cv2.putText(frame, summary_text, (self.frame_width - 250, self.frame_height + 25), 
                   self.font, self.small_font_scale, self.colors['highlight'], 
                   self.small_font_thickness)
        
        y_offset = 50
        for obj_class, track_ids in class_detections.items():
            color = self.colors.get(obj_class, self.colors['text'])
            # Show class count and track IDs
            track_ids_str = ','.join([str(tid) for tid in track_ids[:5]])  # Show max 5 IDs
            if len(track_ids) > 5:
                track_ids_str += "..."
            count_text = f"{obj_class}({len(track_ids)}): ID[{track_ids_str}]"
            cv2.putText(frame, count_text, (self.frame_width - 250, self.frame_height + y_offset), 
                       self.font, self.small_font_scale, color, 
                       self.small_font_thickness)
            y_offset += 20
        
        return frame
    
    def add_category_ids_overlay(self, frame: np.ndarray, current_detections: List[Dict], 
                                sam_tracker=None) -> np.ndarray:
        """
        Add category IDs overlay to the main video frame - FIXED for better visibility.
        
        Args:
            frame: Original video frame (not the extended overlay frame)
            current_detections: Current frame detections
            sam_tracker: SAM tracker instance for category ID retrieval
            
        Returns:
            Frame with category IDs overlaid
        """
        if not sam_tracker:
            return frame
            
        # Add semi-transparent background for better text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 200), (0, 0, 0), -1)  # Black background
        cv2.rectangle(overlay, (frame.shape[1] - 250, 5), (frame.shape[1] - 5, 200), (0, 0, 0), -1)  # Black background right
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)  # Blend for transparency
        
        # Left side: Currently entering objects (from recent line crossings)
        recent_line_crossings = getattr(sam_tracker, 'line_crossings', [])
        if recent_line_crossings:
            # Get recent crossings (last 10 frames worth)
            recent_crossings = [c for c in recent_line_crossings[-10:] if c.get('direction') == 'IN']
            
            if recent_crossings:
                cv2.putText(frame, "Currently Entering:", (10, 25), 
                           self.font, self.small_font_scale, self.colors['highlight'], 
                           self.small_font_thickness)
                
                y_offset = 45
                shown_ids = set()  # Prevent duplicates
                for crossing in recent_crossings[-5:]:  # Show last 5 unique
                    obj_class = crossing.get('class', 'unknown')
                    track_id = crossing.get('track_id', 'N/A')
                    
                    # Get category ID
                    category_id = 'N/A'
                    if hasattr(sam_tracker, 'category_ids') and track_id in sam_tracker.category_ids:
                        category_id = sam_tracker.category_ids[track_id]
                    elif hasattr(sam_tracker, '_get_category_id') and track_id != 'N/A':
                        try:
                            category_id = sam_tracker._get_category_id(int(track_id), obj_class)
                        except:
                            pass
                    
                    # Skip if already shown
                    if category_id in shown_ids or category_id == 'N/A':
                        continue
                    shown_ids.add(category_id)
                    
                    color = self.colors.get(obj_class, self.colors['text'])
                    text = f"{category_id} ({obj_class})"
                    cv2.putText(frame, text, (15, y_offset), 
                               self.font, self.small_font_scale, color, 
                               self.small_font_thickness)
                    y_offset += 22
            else:
                cv2.putText(frame, "Currently Entering: None", (10, 25), 
                           self.font, self.small_font_scale, self.colors['text'], 
                           self.small_font_thickness)
        else:
            cv2.putText(frame, "Currently Entering: N/A", (10, 25), 
                       self.font, self.small_font_scale, self.colors['text'], 
                       self.small_font_thickness)
        
        # Right side: All detected objects with category IDs - IMPROVED layout
        if hasattr(sam_tracker, 'category_ids') and sam_tracker.category_ids:
            cv2.putText(frame, "All Detected:", (frame.shape[1] - 240, 25), 
                       self.font, self.small_font_scale, self.colors['highlight'], 
                       self.small_font_thickness)
            
            # Group by object class using locked class information
            class_objects = {}
            for track_id, category_id in sam_tracker.category_ids.items():
                # Use locked class from sam_tracker.id_to_class
                obj_class = sam_tracker.id_to_class.get(track_id, 'unknown')
                
                # If still unknown, try to infer from category_id prefix
                if obj_class == 'unknown':
                    if category_id.startswith('P'):
                        obj_class = 'person'
                    elif category_id.startswith('B'):
                        obj_class = 'backpack'
                    elif category_id.startswith('H'):
                        obj_class = 'handbag'
                    elif category_id.startswith('S'):
                        obj_class = 'suitcase'
                
                if obj_class not in class_objects:
                    class_objects[obj_class] = []
                class_objects[obj_class].append(category_id)
            
            y_offset = 45
            for obj_class in ['person', 'backpack', 'handbag', 'suitcase']:  # Consistent order
                if obj_class in class_objects:
                    category_ids = class_objects[obj_class]
                    color = self.colors.get(obj_class, self.colors['text'])
                    ids_text = ','.join(sorted(category_ids))
                    text = f"{obj_class}: {ids_text}"
                    cv2.putText(frame, text, (frame.shape[1] - 235, y_offset), 
                               self.font, self.small_font_scale, color, 
                               self.small_font_thickness)
                    y_offset += 22
        
        return frame
    
    def add_performance_info(self, frame: np.ndarray, fps: float, 
                           processing_time: float = None) -> np.ndarray:
        """
        Add performance information to overlay.
        
        Args:
            frame: Frame with overlay area
            fps: Current FPS
            processing_time: Processing time for current frame
            
        Returns:
            Frame with performance info added
        """
        # Performance info at bottom right
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (self.frame_width - 150, self.total_height - 20), 
                   self.font, self.small_font_scale, self.colors['success'], 
                   self.small_font_thickness)
        
        if processing_time:
            time_text = f"Time: {processing_time:.3f}s"
            cv2.putText(frame, time_text, (self.frame_width - 150, self.total_height - 5), 
                       self.font, self.small_font_scale, self.colors['success'], 
                       self.small_font_thickness)
        
        return frame
    
    def add_timestamp(self, frame: np.ndarray, timestamp: str) -> np.ndarray:
        """
        Add timestamp to overlay.
        
        Args:
            frame: Frame with overlay area
            timestamp: Timestamp string
            
        Returns:
            Frame with timestamp added
        """
        cv2.putText(frame, timestamp, (10, self.total_height - 5), 
                   self.font, self.small_font_scale, self.colors['text'], 
                   self.small_font_thickness)
        
        return frame
    
    def create_complete_overlay(self, original_frame: np.ndarray, 
                              line_crossings: List[Dict], 
                              detections: List[Dict],
                              current_frame: int, 
                              total_frames: int,
                              fps: float = 30.0,
                              processing_time: float = None,
                              timestamp: str = None,
                              sam_tracker=None) -> np.ndarray:
        """
        Create complete overlay with all information.
        
        Args:
            original_frame: Original frame
            line_crossings: Line crossing detections
            detections: Object detections
            current_frame: Current frame number
            total_frames: Total frames
            fps: Current FPS
            processing_time: Processing time
            timestamp: Timestamp
            
        Returns:
            Frame with complete overlay
        """
        # Create base overlay frame
        overlay_frame = self.create_overlay_frame(original_frame)
        
        # Add line crossing information
        overlay_frame = self.add_line_info(overlay_frame, line_crossings, current_frame, total_frames, sam_tracker)
        
        # Add detection summary
        overlay_frame = self.add_detection_summary(overlay_frame, detections, sam_tracker)
        
        # Add performance info
        overlay_frame = self.add_performance_info(overlay_frame, fps, processing_time)
        
        # Add timestamp if provided
        if timestamp:
            overlay_frame = self.add_timestamp(overlay_frame, timestamp)
        
        return overlay_frame
    
    def draw_line_indicators(self, frame: np.ndarray, line_crossings: List[Dict]) -> np.ndarray:
        """
        Draw visual indicators for line crossings on the main frame.
        
        Args:
            frame: Frame to draw on
            line_crossings: Line crossing detections
            
        Returns:
            Frame with line indicators
        """
        for crossing in line_crossings:
            position = crossing.get('position', (0, 0))
            obj_class = crossing.get('class', 'unknown')
            direction = crossing.get('direction', 'unknown')
            
            # Draw circle at crossing position
            color = self.colors.get(obj_class, self.colors['highlight'])
            cv2.circle(frame, (int(position[0]), int(position[1])), 10, color, -1)
            
            # Draw direction arrow
            if direction == 'IN':
                arrow_color = self.colors['success']
            else:
                arrow_color = self.colors['warning']
            
            # Draw arrow
            arrow_start = (int(position[0]), int(position[1]))
            if direction == 'IN':
                arrow_end = (int(position[0]) - 20, int(position[1]))
            else:
                arrow_end = (int(position[0]) + 20, int(position[1]))
            
            cv2.arrowedLine(frame, arrow_start, arrow_end, arrow_color, 3)
            
            # Add text label with category ID
            track_id = crossing.get('track_id', 'N/A')
            category_id = crossing.get('category_id', track_id)
            label = f"{obj_class} {category_id} {direction}"
            cv2.putText(frame, label, (int(position[0]) + 15, int(position[1])), 
                       self.font, 0.5, color, 1)
        
        return frame 