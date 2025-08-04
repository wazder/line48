#!/usr/bin/env python3
"""
Run SAM-only analysis for video processing
Pure SAM-based approach without YOLO dependency
"""

# 🎬 FRAME RANGE CONFIGURATION
START_FRAME = 908        # Starting frame
END_FRAME = 1200         # Ending frame (None = until end of video)

import os
import sys
import cv2
import numpy as np
from datetime import datetime
import time
import torch

from sam_only_utils import SAMOnlyLineLogic, download_sam_model
from sam_frame_logic import SAMSegmentTracker
from frame_overlay import FrameOverlay
from line_config import LINE_POINTS, LINE_HEIGHT, BASE_X, LINE_SPACING
from config import LINE_END_Y
import supervision as sv
from supervision import VideoInfo

def run_sam_only_analysis(video_path, output_dir="outputs", start_frame=0, end_frame=None):
    """
    Process video with SAM-only analysis for specified frame range
    """
    print("🎬 SAM-Only Video Analysis")
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
    print("\n🤖 Loading SAM-only model...")
    sam_model = "vit_b"  # Default SAM model
    sam_checkpoint = download_sam_model(sam_model)
    
    # Get video info
    video_info = VideoInfo.from_video_path(video_path)
    total_frames = int(video_info.total_frames) if hasattr(video_info, 'total_frames') else "Unknown"
    duration = video_info.total_frames / video_info.fps if hasattr(video_info, 'total_frames') else "Unknown"
    print(f"📊 Video info: {video_info.width}x{video_info.height}, {video_info.fps} FPS")
    print(f"📊 Total frames: {total_frames}, Duration: {duration} seconds")
    
    # Create line zones
    LINES = []
    LINE_IDS = ["LeftMost", "Left", "Center", "Right", "RightMost"]
    for i, point in enumerate(LINE_POINTS):
        start_point = sv.Point(point.x, point.y)
        end_point = sv.Point(point.x, LINE_END_Y)
        line_zone = sv.LineZone(start=start_point, end=end_point)
        LINES.append(line_zone)
        print(f"   Line {i+1} ({LINE_IDS[i]}): ({point.x}, {point.y}) -> ({point.x}, {LINE_END_Y})")
    
    # Extract x-positions for SAM tracker
    LINE_X_POSITIONS = [point.x for point in LINE_POINTS]
    
    # Initialize SAM segment tracker
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
    
    # Initialize SAM-only logic
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sam_logic = SAMOnlyLineLogic(sam_model_type="vit_b", sam_checkpoint=sam_checkpoint, device=device)
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if end_frame is not None:
        output_path = os.path.join(output_dir, f"{video_name}_sam_only_f{start_frame}-{end_frame}_{timestamp}.mp4")
    else:
        output_path = os.path.join(output_dir, f"{video_name}_sam_only_f{start_frame}-end_{timestamp}.mp4")
    
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
        print(f"\n🔄 Processing frames {start_frame}-{end_frame-1} with SAM-only...")
    else:
        print(f"\n🔄 Processing from frame {start_frame} to end of video with SAM-only...")
    
    while end_frame is None or frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video reached at frame {frame_count}")
            break
        
        # Detect and segment with SAM-only
        segmented_frame, detections = sam_logic.detect_and_segment(frame, frame_count)
        
        # Process line crossings
        valid_detections = []
        if detections:
            for det in detections:
                if det.get('mask') is not None:
                    # Add category-specific ID
                    track_id = det.get('track_id', 'N/A')
                    if track_id != 'N/A':
                        obj_class = det['class']
                        category_id = sam_tracker._get_category_id(track_id, obj_class)
                        det['category_id'] = category_id
                    valid_detections.append(det)
        
        crossings = sam_tracker.update(frame_count, valid_detections)
        
        # Update frame with correct category IDs after processing
        frame = segmented_frame.copy()
        
        # Redraw bounding boxes with correct category IDs
        if valid_detections:
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
                        
                        # Add label with category ID
                        category_id = det.get('category_id', det.get('track_id', 'N/A'))
                        confidence = det.get('confidence', 0.0)
                        
                        label = f"{category_id} ({confidence:.2f})"
                        
                        # Find non-overlapping position
                        label_x, label_y = x1, y1 - 10
                        for used_x, used_y in used_positions:
                            if abs(label_x - used_x) < 80 and abs(label_y - used_y) < 20:
                                label_y -= 20
                        
                        # Add background for readability
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
        
        # Add category IDs overlay to the main video frame
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
        
        # Progress update
        if frame_count % 50 == 0:  # More frequent updates for SAM-only
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
    
    # Run SAM-only analysis
    output_path = run_sam_only_analysis(video_path, start_frame=START_FRAME, end_frame=END_FRAME)
    
    if output_path:
        print(f"\n🎉 SAM-only analysis completed successfully!")
        print(f"📺 Output saved to: {output_path}")
    else:
        print("❌ Processing failed")