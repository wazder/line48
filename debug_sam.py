#!/usr/bin/env python3
"""Debug SAM - Check YOLO detections and line crossings"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from src.config import LINE_POINTS

def debug_sam():
    print("🔍 SAM Debug - YOLO Detections and Line Analysis")
    print("=" * 60)
    
    # Load video
    video_path = "videos/short_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Properties:")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    
    # Load YOLO
    model = YOLO("yolo11n.pt")
    
    # Check line positions
    print(f"\n📏 Line Configuration:")
    for i, point in enumerate(LINE_POINTS):
        print(f"   Line {i+1}: x={point.x}, y={point.y}")
        if point.x < 0 or point.x > frame_width:
            print(f"   ⚠️ Line {i+1} is outside video bounds!")
    
    # Process first few frames
    frame_count = 0
    max_frames = 10
    previous_positions = {}  # Track previous positions for crossing detection
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"\n🟢 Frame {frame_count}:")
        
        # YOLO detection
        results = model.track(frame, verbose=False, persist=True)
        
        if len(results) == 0 or results[0].boxes is None:
            print("   ❌ No detections")
            frame_count += 1
            continue
        
        detections = results[0].boxes
        print(f"   📊 YOLO detections: {len(detections)}")
        
        # Analyze each detection
        for i, box in enumerate(detections):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None and len(box.id) > 0 else i
            
            # Get bbox center
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            print(f"     {i+1}. {class_name} (ID:{track_id}) - Conf: {confidence:.3f}")
            print(f"        Position: ({center_x:.1f}, {center_y:.1f})")
            
            # Check line crossings
            if track_id in previous_positions:
                prev_x = previous_positions[track_id]
                for line_idx, line_point in enumerate(LINE_POINTS):
                    line_x = line_point.x
                    
                    # Check if crossed this line
                    if (prev_x < line_x < center_x) or (center_x < line_x < prev_x):
                        direction = "→" if prev_x < center_x else "←"
                        print(f"        🎯 CROSSED Line {line_idx+1} ({direction})")
                        print(f"           Movement: {prev_x:.1f} → {center_x:.1f} (line at {line_x})")
                    elif abs(center_x - line_x) < 50:  # Near line
                        print(f"        📍 Near line {line_idx+1} (distance: {abs(center_x - line_x):.1f}px)")
            
            # Store current position for next frame
            previous_positions[track_id] = center_x
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Debug completed - analyzed {frame_count} frames")

if __name__ == "__main__":
    debug_sam() 