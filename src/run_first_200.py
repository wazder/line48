#!/usr/bin/env python3
"""
Run SAM analysis for full video
"""

# 🎬 QUICK FRAME CONFIGURATION - Change this value to process different amounts of frames
# Set to None for entire video, or any number for specific frame count
FRAME_LIMIT = 2000  # Examples: None (full video), 800, 1000, 2000, etc.
# APTAL OLMA BURDAN DEGISTIR !!!!!!!!! >*<

import os
import sys
import cv2
import numpy as np
from datetime import datetime
import time

# No need for path manipulation since we're in src/

from sam_utils import download_sam_model, load_sam_model
from sam_frame_logic import SAMSegmentTracker
from frame_overlay import FrameOverlay
from line_config import LINE_POINTS, LINE_HEIGHT, BASE_X, LINE_SPACING
import supervision as sv
from supervision import VideoInfo

def run_sam_full_video_analysis(video_path, output_dir="outputs", max_frames=None):
    """
    Process entire video with SAM analysis
    """
    print("🎬 SAM Full Video Analysis")
    print("=" * 50)
    print(f"📹 Video: {video_path}")
    if max_frames is not None:
        print(f"🎯 Max frames: {max_frames}")
        print(f"🎯 Estimated duration: {max_frames/29:.1f} seconds at 29 FPS")
    else:
        print(f"🎯 Processing entire video")
    
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
        print(f"   {line_names[i]}: ({x}, {y}) -> ({x}, {LINE_HEIGHT})")
    
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
        end_point = sv.Point(point.x, LINE_HEIGHT)
        line_zone = sv.LineZone(start=start_point, end=end_point)
        LINES.append(line_zone)
        print(f"   Line {i+1} ({LINE_IDS[i]}): ({point.x}, {point.y}) -> ({point.x}, {LINE_HEIGHT})")
    
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
    
    # Initialize SAM logic once (not per frame)
    from sam_utils import SAMLineLogic
    sam_logic = SAMLineLogic(sam_model_type="vit_b", sam_checkpoint=sam_checkpoint)
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if max_frames is not None:
        output_path = os.path.join(output_dir, f"{video_name}_sam_{max_frames}frames_{timestamp}.mp4")
    else:
        output_path = os.path.join(output_dir, f"{video_name}_sam_full_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.width, video_info.height + 150))
    
    # Process frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()
    processed_frames = 0
    
    if max_frames is not None:
        print(f"\n🔄 Processing frames 0-{max_frames-1} with SAM...")
    else:
        print(f"\n🔄 Processing entire video with SAM...")
    
    while max_frames is None or frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video reached at frame {frame_count}")
            break
        
        # Detect and segment with SAM
        segmented_frame, detections = sam_logic.detect_and_segment(frame, frame_count)
        
        # Process line crossings - ensure detections have proper format
        if detections:
            # Filter detections that have masks and add category IDs
            valid_detections = []
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
        else:
            crossings = []
        
        # Annotate frame - detections is a list, not sv.Detections object
        # Skip annotation for now since detections is in list format
        frame = segmented_frame
        
        # Annotate lines
        for line, line_annotator in zip(LINES, line_annotators):
            frame = line_annotator.annotate(frame, line)
        
        # Add line crossing indicators
        frame = frame_overlay.draw_line_indicators(frame, crossings)
        
        # Add category IDs overlay to the main video frame
        frame = frame_overlay.add_category_ids_overlay(frame, detections, sam_tracker)
        
        # Create overlay with information
        overlay_frame = frame_overlay.create_complete_overlay(
            original_frame=frame,
            line_crossings=crossings,
            detections=detections,
            current_frame=frame_count,
            total_frames=max_frames if max_frames is not None else total_frames,
            fps=video_info.fps,
            processing_time=None,
            timestamp=f"Frame {frame_count}/{max_frames if max_frames is not None else total_frames}",
            sam_tracker=sam_tracker
        )
        
        # Write frame
        out.write(overlay_frame)
        
        # Progress update - reduced frequency for cleaner output
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = processed_frames / elapsed if elapsed > 0 else 0
            if max_frames is not None:
                print(f"🟢 Processed frame {frame_count}/{max_frames} | Speed: {fps_processing:.1f} FPS")
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
    
    # Run SAM analysis with configured frame limit
    output_path = run_sam_full_video_analysis(video_path, max_frames=FRAME_LIMIT)
    
    if output_path:
        print(f"\n🎉 SAM full video analysis completed successfully!")
        print(f"📺 Output saved to: {output_path}")
    else:
        print("❌ Processing failed") 