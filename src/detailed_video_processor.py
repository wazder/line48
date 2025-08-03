"""
Detailed Video Processor with Frame-by-Frame Information Display
Shows detailed detection and tracking information for each frame.
"""

import cv2
import numpy as np
import supervision as sv
from typing import List, Dict, Any
import time

def add_detailed_frame_info(frame: np.ndarray, frame_idx: int, detections: List[Dict], 
                           crossings: List[Dict], fps: float) -> np.ndarray:
    """
    Add detailed frame information to the bottom of the frame.
    
    Args:
        frame: Current video frame
        frame_idx: Current frame index
        detections: List of detections in current frame
        crossings: List of line crossings in current frame
        fps: Video FPS
        
    Returns:
        Frame with detailed information overlay
    """
    height, width = frame.shape[:2]
    
    # Create info panel at bottom (200px height)
    info_height = 200
    info_panel = np.zeros((info_height, width, 3), dtype=np.uint8)
    info_panel[:] = (40, 40, 40)  # Dark gray background
    
    # Calculate timestamp
    timestamp = frame_idx / fps
    minutes = int(timestamp // 60)
    seconds = timestamp % 60
    
    # Header info
    header_text = f"Frame: {frame_idx} | Time: {minutes:02d}:{seconds:05.2f} | FPS: {fps:.1f}"
    cv2.putText(info_panel, header_text, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Detection count
    detection_counts = {}
    for det in detections:
        class_name = det.get('class', 'unknown')
        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
    
    if detection_counts:
        det_text = "Detections: " + ", ".join([f"{cls}: {count}" for cls, count in detection_counts.items()])
        cv2.putText(info_panel, det_text, (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(info_panel, "Detections: None", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    # Line crossings info
    if crossings:
        crossing_text = f"Line Crossings: {len(crossings)}"
        cv2.putText(info_panel, crossing_text, (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Show crossing details
        y_offset = 115
        for i, crossing in enumerate(crossings[:3]):  # Show max 3 crossings
            crossing_detail = f"  {crossing.get('class', 'unknown')} → Line {crossing.get('line_id', '?')} ({crossing.get('direction', '?')})"
            cv2.putText(info_panel, crossing_detail, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
    else:
        cv2.putText(info_panel, "Line Crossings: None", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    # Detection details on the right side
    if detections:
        y_offset = 25
        x_offset = width // 2
        cv2.putText(info_panel, "Detection Details:", (x_offset, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        for i, det in enumerate(detections[:4]):  # Show max 4 detections
            class_name = det.get('class', 'unknown')
            confidence = det.get('confidence', 0.0)
            track_id = det.get('track_id', 'N/A')
            
            detail_text = f"ID:{track_id} {class_name} ({confidence:.2f})"
            cv2.putText(info_panel, detail_text, (x_offset, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
            
            if y_offset > info_height - 10:
                break
    
    # Combine original frame with info panel
    combined_frame = np.vstack([frame, info_panel])
    
    return combined_frame

def process_video_with_detailed_info(source_path: str, target_path: str, 
                                   model, tracker, line_zones, 
                                   confidence: float = 0.25, verbose: bool = True):
    """
    Process video with detailed frame-by-frame information display.
    
    Args:
        source_path: Input video path
        target_path: Output video path
        model: YOLO model
        tracker: FrameBasedTracker instance
        line_zones: Line zones for crossing detection
        confidence: Detection confidence threshold
        verbose: Whether to print verbose output
    """
    
    # Open video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer (with extra height for info panel)
    output_height = frame_height + 200
    
    # Try different codecs for better compatibility
    try:
        # First try H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(target_path, fourcc, fps, (frame_width, output_height))
        
        if not out.isOpened():
            print("⚠️ H.264 codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            target_path = target_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(target_path, fourcc, fps, (frame_width, output_height))
            
        if not out.isOpened():
            print("⚠️ XVID codec failed, trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(target_path, fourcc, fps, (frame_width, output_height))
            
        if not out.isOpened():
            raise ValueError("All video codecs failed")
            
        print(f"✅ Video writer initialized with codec: {fourcc}")
        
    except Exception as e:
        print(f"❌ Video writer setup failed: {e}")
        raise
    
    frame_idx = 0
    start_time = time.time()
    
    print(f"🎬 Processing video with detailed frame info...")
    print(f"   Input: {source_path}")
    print(f"   Output: {target_path}")
    print(f"   Resolution: {frame_width}x{frame_height} → {frame_width}x{output_height}")
    print(f"   Total frames: {total_frames}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection
            results = model(frame, conf=confidence, verbose=False)
            
            # Convert to supervision detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Filter for target classes
            target_class_ids = [0, 24, 26, 28]  # person, backpack, handbag, suitcase
            mask = np.isin(detections.class_id, target_class_ids)
            detections = detections[mask]
            
            # Track objects
            tracked_detections = tracker.byte_tracker.update_with_detections(detections)
            
            # Prepare detection info for display
            detection_info = []
            for i in range(len(tracked_detections)):
                class_id = tracked_detections.class_id[i]
                confidence_val = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.0
                track_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else 'N/A'
                
                class_name = model.names[class_id] if class_id in model.names else 'unknown'
                
                detection_info.append({
                    'class': class_name,
                    'confidence': confidence_val,
                    'track_id': track_id,
                    'class_id': class_id
                })
            
            # Check line crossings (simplified)
            current_crossings = []
            if hasattr(tracker, 'update_with_detections'):
                # Use tracker's crossing detection
                crossings = tracker.update_with_detections(tracked_detections, frame_idx)
                current_crossings = crossings
            
            # Draw detections on frame
            annotated_frame = frame.copy()
            
            # Draw bounding boxes and labels
            for i, detection in enumerate(detection_info):
                if i < len(tracked_detections.xyxy):
                    x1, y1, x2, y2 = tracked_detections.xyxy[i].astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{detection['track_id']}: {detection['class']} ({detection['confidence']:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), 
                                 (x1+label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw line zones
            for line_zone in line_zones:
                line_zone.annotate(annotated_frame)
            
            # Add detailed frame information
            detailed_frame = add_detailed_frame_info(
                annotated_frame, frame_idx, detection_info, current_crossings, fps
            )
            
            # Write frame
            out.write(detailed_frame)
            
            # Progress update
            if verbose and frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                processing_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames) * 100
                print(f"🟢 Frame {frame_idx}/{total_frames} ({progress:.1f}%) | Speed: {processing_fps:.1f} FPS")
            
            frame_idx += 1
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    finally:
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        avg_fps = frame_idx / processing_time if processing_time > 0 else 0
        
        print(f"\n✅ Processing completed!")
        print(f"   Processed frames: {frame_idx}")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Average speed: {avg_fps:.1f} FPS")
        print(f"   Output saved: {target_path}")

def create_comparison_summary(yolo_results: Dict, sam_results: Dict, output_path: str):
    """
    Create a comparison summary between YOLO and SAM results.
    
    Args:
        yolo_results: Results from YOLO analysis
        sam_results: Results from SAM analysis  
        output_path: Path to save comparison summary
    """
    
    summary = []
    summary.append("# YOLO vs SAM Analysis Comparison")
    summary.append("=" * 50)
    summary.append("")
    
    # Detection accuracy comparison
    summary.append("## Detection Accuracy Comparison")
    summary.append("")
    
    classes = ['person', 'backpack', 'handbag', 'suitcase']
    
    for cls in classes:
        yolo_count = yolo_results.get(cls, {}).get('total', 0)
        sam_count = sam_results.get(cls, {}).get('total', 0)
        
        difference = sam_count - yolo_count
        percentage_diff = (difference / yolo_count * 100) if yolo_count > 0 else 0
        
        summary.append(f"**{cls.capitalize()}:**")
        summary.append(f"  - YOLO: {yolo_count} detections")
        summary.append(f"  - SAM: {sam_count} detections")
        summary.append(f"  - Difference: {difference:+d} ({percentage_diff:+.1f}%)")
        summary.append("")
    
    # Performance comparison
    summary.append("## Performance Comparison")
    summary.append("")
    summary.append("| Metric | YOLO | SAM |")
    summary.append("|--------|------|-----|")
    
    yolo_fps = yolo_results.get('processing_fps', 0)
    sam_fps = sam_results.get('processing_fps', 0)
    
    summary.append(f"| Processing Speed | {yolo_fps:.1f} FPS | {sam_fps:.1f} FPS |")
    summary.append(f"| Accuracy | Standard | Higher |")
    summary.append(f"| Memory Usage | Low | High |")
    summary.append("")
    
    # Save summary
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"📊 Comparison summary saved: {output_path}")