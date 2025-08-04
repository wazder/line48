#!/usr/bin/env python3
"""
Run SAM analysis for full video
"""

# 🎬 FRAME RANGE CONFIGURATION - Change these values to process specific frame ranges
# Set START_FRAME and END_FRAME to process a range, or set END_FRAME to None for entire video from start

START_FRAME = 908        # Starting frame (0 = beginning of video)
END_FRAME = 2600       # Ending frame (None = until end of video)

# Examples:
# START_FRAME = 0, END_FRAME = 1000     -> Process frames 0-999 (first 1000 frames)
# START_FRAME = 500, END_FRAME = 1500   -> Process frames 500-1499 (middle 1000 frames)  
# START_FRAME = 1000, END_FRAME = None  -> Process from frame 1000 to end of video
# START_FRAME = 0, END_FRAME = None     -> Process entire video

import os
import sys
import cv2
import numpy as np
from datetime import datetime
import time
import torch

# No need for path manipulation since we're in src/

from sam_utils import download_sam_model, load_sam_model
from sam_frame_logic import SAMSegmentTracker
from frame_overlay import FrameOverlay
from line_config import LINE_POINTS, LINE_HEIGHT, BASE_X, LINE_SPACING
from config import LINE_END_Y
import supervision as sv
from supervision import VideoInfo

def run_sam_video_analysis(video_path, output_dir="outputs", start_frame=0, end_frame=None):
    """
    Process video with SAM analysis for specified frame range
    """
    print("🎬 SAM Video Analysis")
    print("=" * 50)
    print(f"📹 Video: {video_path}")
    
    # Calculate frame range
    total_frames = end_frame - start_frame if end_frame is not None else "Unknown"
    
    print(f"🎯 Start frame: {start_frame}")
    if end_frame is not None:
        print(f"🎯 End frame: {end_frame}")
        print(f"🎯 Total frames to process: {total_frames}")
        print(f"🎯 Estimated duration: {total_frames/30:.1f} seconds at 30 FPS")
    else:
        print(f"🎯 End frame: End of video")
        print(f"🎯 Processing from frame {start_frame} to end of video")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Show line configuration
    print(f"\n📏 Line Configuration:")
    print(f"   BASE_X: {BASE_X}")
    print(f"   LINE_SPACING: {LINE_SPACING}")
    print(f"   LINE_HEIGHT: {LINE_HEIGHT}")
    
    print(f"\n📍 Line Positions:")
    line_names = ["LeftMost", "Left", "Center", "Right", "RightMost"]
    for i, point in enumerate(LINE_POINTS):
        x, y = point.x, point.y
        print(f"   {line_names[i]}: ({x}, {y}) -> ({x}, {LINE_END_Y})")
    
    # Load SAM model
    print("\n🤖 Loading SAM model...")
    sam_model = "vit_b"  # Default SAM model
    sam_checkpoint = download_sam_model(sam_model)
    sam_predictor = load_sam_model(sam_checkpoint)
    print(f"✅ SAM model loaded successfully")
    
    # Get video info
    video_info = VideoInfo.from_video_path(video_path)
    total_frames = int(video_info.total_frames) if hasattr(video_info, 'total_frames') else "Unknown"
    duration = video_info.total_frames / video_info.fps if hasattr(video_info, 'total_frames') else "Unknown"
    print(f"📊 Video info: {video_info.width}x{video_info.height}, {video_info.fps} FPS")
    print(f"📊 Total frames: {total_frames}, Duration: {duration} seconds")
    
    # LINE_POINTS are already sv.Point objects
    LINE_IDS = ["LeftMost", "Left", "Center", "Right", "RightMost"]
    
    # Create line zones
    LINES = []
    for i, point in enumerate(LINE_POINTS):
        start_point = sv.Point(point.x, point.y)
        end_point = sv.Point(point.x, LINE_END_Y)
        line_zone = sv.LineZone(start=start_point, end=end_point)
        LINES.append(line_zone)
        print(f"   Line {i+1} ({LINE_IDS[i]}): ({point.x}, {point.y}) -> ({point.x}, {LINE_END_Y})")
    
    # Extract x-positions for SAM tracker
    LINE_X_POSITIONS = [point.x for point in LINE_POINTS]
    
    # Initialize SAM segment tracker with explicit line positions
    sam_tracker = SAMSegmentTracker(lines=LINES, fps=video_info.fps, line_x_positions=LINE_X_POSITIONS)
    line_annotators = [
        sv.LineZoneAnnotator(
            display_in_count=False,
            display_out_count=False,
            text_thickness=2,
            text_scale=1.0
        )
        for _ in LINE_IDS
    ]
    
    # Initialize frame overlay
    frame_overlay = FrameOverlay(video_info.height, video_info.width, overlay_height=150)
    
    # Initialize SAM logic once (not per frame) - Force GPU usage
    from sam_utils import SAMLineLogic
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sam_logic = SAMLineLogic(sam_model_type="vit_b", sam_checkpoint=sam_checkpoint, device=device)
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if end_frame is not None:
        output_path = os.path.join(output_dir, f"{video_name}_sam_f{start_frame}-{end_frame}_{timestamp}.mp4")
    else:
        output_path = os.path.join(output_dir, f"{video_name}_sam_f{start_frame}-end_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.width, video_info.height + 150))
    
    # Process frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()
    processed_frames = 0
    
    # Skip to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        print(f"\n⏭️ Skipping to frame {start_frame}")
    
    if end_frame is not None:
        print(f"\n🔄 Processing frames {start_frame}-{end_frame-1} with SAM...")
    else:
        print(f"\n🔄 Processing from frame {start_frame} to end of video with SAM...")
    
    while end_frame is None or frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video reached at frame {frame_count}")
            break
        
        # Detect and segment with SAM
        segmented_frame, detections = sam_logic.detect_and_segment(frame, frame_count)
        
        # Process line crossings - ensure detections have proper format
        valid_detections = []
        if detections:
            # Filter detections that have masks and add category IDs
            for det in detections:
                if det.get('mask') is not None:
                    # Add category-specific ID using locked class if available
                    track_id = det.get('track_id', 'N/A')
                    if track_id != 'N/A':
                        # Use locked class if track is already known, otherwise use detected class
                        obj_class = det['class']
                        if track_id in sam_tracker.id_to_class:
                            locked_class = sam_tracker.id_to_class[track_id]
                            if locked_class != obj_class:
                                print(f"🔄 Using locked class: Track {track_id} detected as {obj_class} but locked as {locked_class}")
                                obj_class = locked_class
                        
                        category_id = sam_tracker._get_category_id(track_id, obj_class)
                        det['category_id'] = category_id
                        det['class'] = obj_class  # Use locked class for further processing
                    valid_detections.append(det)
            
        crossings = sam_tracker.update(frame_count, valid_detections)
        
        # Update frame with correct category IDs after processing
        frame = segmented_frame.copy()
        
        # Redraw bounding boxes with correct category IDs - IMPROVED text positioning
        if valid_detections:
            # Track label positions to prevent overlap
            used_positions = []
            
            for i, det in enumerate(valid_detections):
                if det.get('mask') is not None:
                    bbox = det.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        
                        # Draw bounding box with class-specific color
                        obj_class = det.get('class', 'unknown')
                        box_colors = {
                            'person': (255, 150, 100),     # Light blue
                            'backpack': (0, 255, 0),       # Green
                            'handbag': (255, 0, 255),      # Magenta
                            'suitcase': (0, 255, 255)      # Yellow
                        }
                        box_color = box_colors.get(obj_class, (0, 255, 0))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Add label with correct category ID - NO overlap
                        category_id = det.get('category_id', det.get('track_id', 'N/A'))
                        confidence = det.get('confidence', 0.0)
                        
                        # Simplified label - only category ID and confidence
                        label = f"{category_id} ({confidence:.2f})"
                        
                        # Find non-overlapping position for text
                        label_x, label_y = x1, y1 - 10
                        
                        # Check for overlap and adjust position
                        for used_x, used_y in used_positions:
                            if abs(label_x - used_x) < 80 and abs(label_y - used_y) < 20:
                                label_y -= 20  # Move up to avoid overlap
                        
                        # Add background for better readability
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (label_x, label_y - text_height - 2), 
                                    (label_x + text_width, label_y + 2), (0, 0, 0), -1)
                        
                        cv2.putText(frame, label, (label_x, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                        
                        used_positions.append((label_x, label_y))
        
        # Annotate lines
        for line, line_annotator in zip(LINES, line_annotators):
            frame = line_annotator.annotate(frame, line)
        
        # Add line crossing indicators
        frame = frame_overlay.draw_line_indicators(frame, crossings)
        
        # Add category IDs overlay to the main video frame - use valid_detections for consistency
        frame = frame_overlay.add_category_ids_overlay(frame, valid_detections, sam_tracker)
        
        # Create overlay with information
        overlay_frame = frame_overlay.create_complete_overlay(
            original_frame=frame,
            line_crossings=crossings,
            detections=detections,
            current_frame=frame_count,
            total_frames=end_frame if end_frame is not None else "End",
            fps=video_info.fps,
            processing_time=None,
            timestamp=f"Frame {frame_count}/{end_frame if end_frame is not None else 'End'}",
            sam_tracker=sam_tracker
        )
        
        # Write frame
        out.write(overlay_frame)
        
        # Progress update - reduced frequency for cleaner output
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = processed_frames / elapsed if elapsed > 0 else 0
            if end_frame is not None:
                progress = ((frame_count - start_frame) / (end_frame - start_frame)) * 100
                print(f"🟢 Processed frame {frame_count}/{end_frame} ({progress:.1f}%) | Speed: {fps_processing:.1f} FPS")
            else:
                print(f"🟢 Processed frame {frame_count} | Speed: {fps_processing:.1f} FPS")
        
        frame_count += 1
        processed_frames += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get results
    crossing_summary = sam_tracker.get_crossing_summary()
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Output video: {output_path}")
    print(f"📊 Processed frames: {frame_count}")
    
    # Print results
    print("\n📈 Detection Results:")
    print(crossing_summary)
    
    return output_path

if __name__ == "__main__":
    # Use default video path
    video_path = "../videos/new_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    # Run SAM analysis with configured frame range
    output_path = run_sam_video_analysis(video_path, start_frame=START_FRAME, end_frame=END_FRAME)
    
    if output_path:
        print(f"\n🎉 SAM full video analysis completed successfully!")
        print(f"📺 Output saved to: {output_path}")
    else:
        print("❌ Processing failed") 