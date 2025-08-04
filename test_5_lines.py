#!/usr/bin/env python3
"""
Test script for 5-line system
Process first 200 frames with new line configuration
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from utils import load_model
from frame_logic import FrameBasedTracker
from frame_overlay import FrameOverlay
from line_config import LINE_POINTS, LINE_HEIGHT, BASE_X, LINE_SPACING
import supervision as sv
from supervision import VideoInfo

def process_with_5_lines(video_path, output_dir="outputs", max_frames=200):
    """
    Process video with 5-line system
    """
    print("🎬 5-Line System Analysis")
    print("=" * 50)
    print(f"📹 Video: {video_path}")
    print(f"🎯 Max frames: {max_frames}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Show line configuration
    print(f"\n📏 Line Configuration:")
    print(f"   BASE_X: {BASE_X}")
    print(f"   LINE_SPACING: {LINE_SPACING}")
    print(f"   LINE_HEIGHT: {LINE_HEIGHT}")
    
    print(f"\n📍 Line Positions:")
    line_names = ["LeftMost", "Left", "Center", "Right", "RightMost"]
    for i, (x, y) in enumerate(LINE_POINTS):
        print(f"   {line_names[i]}: ({x}, {y}) -> ({x}, {LINE_HEIGHT})")
    
    # Load model
    print("\n🤖 Loading YOLO model...")
    model = load_model()
    model.conf = 0.25
    model.iou = 0.45
    model.imgsz = 1024
    
    # Initialize tracker
    tracker = FrameBasedTracker()
    
    # Get video info
    video_info = VideoInfo.from_video_path(video_path)
    print(f"📊 Video info: {video_info.width}x{video_info.height}, {video_info.fps} FPS")
    
    # Convert line positions to sv.Point objects
    LINE_POINTS_SV = [sv.Point(x, y) for x, y in LINE_POINTS]
    LINE_IDS = ["LeftMost", "Left", "Center", "Right", "RightMost"]
    
    # Create line zones
    LINES = [sv.LineZone(start=p, end=sv.Point(p.x, LINE_HEIGHT)) for p in LINE_POINTS_SV]
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
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_5_lines_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.width, video_info.height + 150))
    
    # Process frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print(f"\n🔄 Processing frames 0-{max_frames-1} with 5-line system...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video reached at frame {frame_count}")
            break
        
        # Run inference
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter to target classes
        COCO_NAMES = model.model.names
        SELECTED_CLASSES = ["person", "backpack", "handbag", "suitcase"]
        SELECTED_CLASS_IDS = [k for k, v in COCO_NAMES.items() if v in SELECTED_CLASSES]
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        
        # Track objects
        detections = tracker.byte_tracker.update_with_detections(detections)
        
        # Update object presence
        tracker.update_object_presence(detections, frame_count, COCO_NAMES)
        
        # Process line crossings
        crossings = tracker.process_line_crossing(detections, frame_count, LINES, LINE_IDS, COCO_NAMES)
        
        # Annotate frame
        frame = sv.BoxAnnotator().annotate(frame, detections)
        frame = sv.LabelAnnotator().annotate(frame, detections)
        
        # Annotate lines
        for line, line_annotator in zip(LINES, line_annotators):
            frame = line_annotator.annotate(frame, line)
        
        # Add line crossing indicators
        frame = frame_overlay.draw_line_indicators(frame, crossings)
        
        # Create overlay with information
        overlay_frame = frame_overlay.create_complete_overlay(
            original_frame=frame,
            line_crossings=crossings,
            detections=[{
                'class': COCO_NAMES.get(det.class_id, 'unknown'),
                'confidence': det.confidence,
                'track_id': det.tracker_id,
                'bbox': det.xyxy[0] if len(det.xyxy) > 0 else [0, 0, 0, 0]
            } for det in detections],
            current_frame=frame_count,
            total_frames=max_frames,
            fps=video_info.fps,
            processing_time=None,
            timestamp=f"Frame {frame_count}/{max_frames}"
        )
        
        # Write frame
        out.write(overlay_frame)
        
        # Progress update
        if frame_count % 50 == 0:
            print(f"🟢 Processed frame {frame_count}/{max_frames}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get results
    results_summary = tracker.get_results_summary()
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Output video: {output_path}")
    print(f"📊 Processed frames: {frame_count}")
    
    # Print results
    print("\n📈 Detection Results:")
    for class_name, counts in results_summary.items():
        print(f"   {class_name}: {counts}")
    
    # Print line crossing results
    print("\n🎯 Line Crossing Results:")
    for line_id in LINE_IDS:
        in_count = tracker.get_line_in_count(line_id)
        out_count = tracker.get_line_out_count(line_id)
        print(f"   {line_id}: IN={in_count}, OUT={out_count}")
    
    return output_path

if __name__ == "__main__":
    # Use the test video
    video_path = "videos/test14-3.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        sys.exit(1)
    
    # Process with 5-line system
    output_path = process_with_5_lines(video_path)
    
    if output_path:
        print(f"\n🎉 5-line system analysis completed successfully!")
        print(f"📺 Output saved to: {output_path}")
    else:
        print("❌ Processing failed") 