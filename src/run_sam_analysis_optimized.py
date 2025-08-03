#!/usr/bin/env python3
"""
Optimized SAM LineLogic Analysis Runner
Uses stricter parameters to get more realistic results similar to YOLO.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

# Import SAM modules
from sam_utils import SAMLineLogic, download_sam_model
from sam_frame_logic import SAMSegmentTracker
import config
import supervision as sv

def run_optimized_sam_analysis(video_path: str, sam_model: str = "vit_b", detailed_info: bool = False):
    """
    Run optimized SAM analysis with realistic parameters.
    
    Args:
        video_path: Path to input video
        sam_model: SAM model type
        detailed_info: Show detailed frame info
    """
    
    print("🔬 Optimized SAM + LineLogic Analysis Starting...")
    print("=" * 60)
    
    # Check video file
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Setup output paths
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_video_path = f"../outputs/{video_name}_sam_optimized_{timestamp}.mp4"
    log_csv_path = f"../logs/{video_name}_sam_optimized_log_{timestamp}.csv"
    results_csv_path = f"../logs/{video_name}_sam_optimized_results_{timestamp}.csv"
    
    print(f"🎬 Processing: {os.path.basename(video_path)}")
    print(f"📁 Output: {output_video_path}")
    print(f"📝 Log: {log_csv_path}")
    print(f"📊 Results: {results_csv_path}")
    
    # Download SAM model if needed
    try:
        sam_checkpoint = download_sam_model(sam_model)
    except Exception as e:
        print(f"❌ Failed to download SAM model: {e}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Properties:")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Initialize SAM + LineLogic with STRICT parameters
    try:
        sam_logic = SAMLineLogic(
            sam_model_type=sam_model,
            sam_checkpoint=sam_checkpoint,
            yolo_model="yolo11n.pt",
            device="cuda:0"
        )
    except Exception as e:
        print(f"❌ Failed to initialize SAM: {e}")
        return
    
    # Setup line configuration
    lines = []
    for point in config.LINE_POINTS:
        lines.append(sv.Point(point.x, frame_height))
    
    # Initialize segment tracker with VERY STRICT parameters
    segment_tracker = SAMSegmentTracker(
        lines=lines,
        fps=fps,
        min_safe_time=3.0,      # VERY strict: 3 seconds minimum
        min_uncertain_time=1.5,  # VERY strict: 1.5 seconds
        min_very_brief_time=0.8  # VERY strict: 0.8 seconds
    )
    
    # Setup video writer
    output_height = frame_height + 200 if detailed_info else frame_height
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, output_height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_path = output_video_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, output_height))
        
        print(f"✅ Video writer initialized")
    except Exception as e:
        print(f"❌ Video writer failed: {e}")
        return
    
    # Processing variables
    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    detection_log = []
    
    print(f"\\n🔥 Starting OPTIMIZED SAM processing...")
    print(f"   SAM Model: {sam_model}")
    print(f"   STRICT Parameters: min_safe=3.0s, confidence=0.5, mask_area>1000px")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # SAM detection and segmentation with HIGHER confidence
            segmented_frame, detections = sam_logic.detect_and_segment(frame)
            
            # Additional filtering for SAM results
            filtered_detections = []
            for detection in detections:
                # Only keep high-confidence detections with reasonable mask sizes
                if (detection.get('confidence', 0) > 0.5 and 
                    detection.get('mask') is not None and
                    np.sum(detection['mask']) > 2000):  # Larger minimum mask area
                    filtered_detections.append(detection)
            
            # Update segment tracker with filtered detections
            crossings = segment_tracker.update(frame_idx, filtered_detections)
            
            # Log detections
            for detection in filtered_detections:
                log_entry = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'bbox_x1': detection['bbox'][0],
                    'bbox_y1': detection['bbox'][1],
                    'bbox_x2': detection['bbox'][2],
                    'bbox_y3': detection['bbox'][3],
                    'mask_area': np.sum(detection['mask']) if detection['mask'] is not None else 0,
                    'mask_score': detection.get('mask_score', 0.0)
                }
                detection_log.append(log_entry)
            
            # Draw lines on frame
            for i, line_point in enumerate(config.LINE_POINTS):
                cv2.line(segmented_frame, 
                        (line_point.x, 0), 
                        (line_point.x, frame_height),
                        (0, 255, 255), 2)
                cv2.putText(segmented_frame, f"L{i+1}", 
                           (line_point.x + 5, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add frame info
            if detailed_info:
                from detailed_video_processor import add_detailed_frame_info
                
                detection_info = []
                for detection in filtered_detections:
                    detection_info.append({
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'track_id': detection.get('track_id', 'N/A'),
                        'mask_area': detection.get('mask_area', 0)
                    })
                
                segmented_frame = add_detailed_frame_info(
                    segmented_frame, frame_idx, detection_info, crossings, fps
                )
            else:
                # Add basic frame info
                info_text = f"Frame: {frame_idx}/{total_frames} | Optimized SAM: {len(filtered_detections)} detections"
                cv2.putText(segmented_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(segmented_frame)
            
            # Progress update
            if processed_frames % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                print(f"🟢 Processed frame {frame_idx} | Speed: {fps_processing:.1f} FPS | Detections: {len(filtered_detections)}")
            
            frame_idx += 1
            processed_frames += 1
        
    except KeyboardInterrupt:
        print("\\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\\n❌ Processing error: {e}")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
    
    # Calculate final statistics
    processing_time = time.time() - start_time
    avg_fps = processed_frames / processing_time if processing_time > 0 else 0
    
    print(f"\\n✅ OPTIMIZED Processing Complete!")
    print(f"   Processed frames: {processed_frames}/{total_frames}")
    print(f"   Processing time: {processing_time:.1f}s")
    print(f"   Average FPS: {avg_fps:.1f}")
    
    # Save detection log
    if detection_log:
        df_log = pd.DataFrame(detection_log)
        df_log.to_csv(log_csv_path, index=False)
        print(f"📝 Detection log saved: {log_csv_path}")
    
    # Generate and save results
    results_summary = segment_tracker.get_crossing_summary()
    print(f"\\n{results_summary}")
    
    # Save results CSV
    crossing_results = segment_tracker.validate_and_count_crossings()
    results_data = []
    
    for obj_class in ['person', 'backpack', 'handbag', 'suitcase']:
        results_data.append({
            'class': obj_class,
            'safe_crossings': crossing_results['safe_crossings'][obj_class],
            'uncertain_crossings': crossing_results['uncertain_crossings'][obj_class],
            'very_brief_crossings': crossing_results['very_brief_crossings'][obj_class],
            'discarded_crossings': crossing_results['discarded_crossings'][obj_class],
            'total_valid_crossings': (crossing_results['safe_crossings'][obj_class] + 
                                    crossing_results['uncertain_crossings'][obj_class] + 
                                    crossing_results['very_brief_crossings'][obj_class]),
            'processing_time': processing_time,
            'avg_fps': avg_fps,
            'sam_model': sam_model,
            'optimization': 'strict_parameters',
            'timestamp': timestamp
        })
    
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(results_csv_path, index=False)
    print(f"📊 Results saved: {results_csv_path}")
    print(f"🎬 Processed video saved: {output_video_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimized SAM + LineLogic Video Analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--sam-model", type=str, default="vit_b", 
                       choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")
    parser.add_argument("--detailed-info", action="store_true",
                       help="Show detailed frame information overlay")
    
    args = parser.parse_args()
    
    print("🎯 Optimized SAM + LineLogic Video Analysis")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"SAM Model: {args.sam_model}")
    print("Optimization: STRICT parameters for realistic results")
    print("=" * 60)
    
    try:
        run_optimized_sam_analysis(args.video, args.sam_model, args.detailed_info)
    except Exception as e:
        print(f"\\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()