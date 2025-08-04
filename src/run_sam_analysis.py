"""
SAM LineLogic Analysis Runner
Main script for running video analysis with Segment Anything Model (SAM) + LineLogic.

Usage:
python run_sam_analysis.py --video "path/to/video.mp4" --sam-model vit_b
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
from frame_overlay import FrameOverlay
import config
import supervision as sv

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="SAM + LineLogic Video Analysis")
    
    # Video parameters
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="../outputs", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="../logs", help="Log directory")
    
    # SAM parameters
    parser.add_argument("--sam-model", type=str, default="vit_b", 
                       choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model type (vit_b=fastest, vit_h=best quality)")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                       help="Path to SAM checkpoint file")
    parser.add_argument("--download-sam", action="store_true",
                       help="Download SAM model if not found")
    
    # YOLO parameters
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt",
                       help="YOLO model for object detection")
    parser.add_argument("--confidence", type=float, default=0.01,  # Much more relaxed
                       help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="YOLO input image size")
    
    # Frame logic parameters
    parser.add_argument("--min-safe-time", type=float, default=0.1,  # Much more relaxed
                       help="Minimum time for safe predictions (seconds)")
    parser.add_argument("--min-uncertain-time", type=float, default=0.05,  # Much more relaxed
                       help="Minimum time for uncertain predictions (seconds)")
    parser.add_argument("--min-very-brief-time", type=float, default=0.01,  # Much more relaxed
                       help="Minimum time for very brief predictions (seconds)")
    
    # Processing parameters
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (for testing)")
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Process every N frames (1=all frames)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run models on")
    
    # Output parameters
    parser.add_argument("--save-video", action="store_true", default=True,
                       help="Save processed video")
    parser.add_argument("--show-video", action="store_true",
                       help="Display video during processing")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument("--detailed-info", action="store_true",
                       help="Show detailed frame-by-frame information overlay")
    
    return parser.parse_args()

def run_sam_analysis(args):
    """Main analysis function."""
    
    print("🚀 SAM + LineLogic Analysis Starting...")
    print("=" * 60)
    
    # Check video file
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    # Setup output paths
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_video_path = os.path.join(args.output_dir, f"{video_name}_sam_processed_{timestamp}.mp4")
    log_csv_path = os.path.join(args.log_dir, f"{video_name}_sam_log_{timestamp}.csv")
    results_csv_path = os.path.join(args.log_dir, f"{video_name}_sam_results_{timestamp}.csv")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"🎬 Processing: {os.path.basename(args.video)}")
    print(f"📁 Output: {output_video_path}")
    print(f"📝 Log: {log_csv_path}")
    print(f"📊 Results: {results_csv_path}")
    
    # Download SAM model if needed
    if args.download_sam or args.sam_checkpoint is None:
        try:
            sam_checkpoint = download_sam_model(args.sam_model)
            args.sam_checkpoint = sam_checkpoint
        except Exception as e:
            print(f"❌ Failed to download SAM model: {e}")
            return
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
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
    
    # Initialize SAM + LineLogic
    try:
        sam_logic = SAMLineLogic(
            sam_model_type=args.sam_model,
            sam_checkpoint=args.sam_checkpoint,
            yolo_model=args.yolo_model,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to initialize SAM: {e}")
        return
    
    # Setup line configuration (using config.py)
    lines = []
    for point in config.LINE_POINTS:
        lines.append(sv.Point(point.x, frame_height))  # Extend to frame height
    
    # Initialize segment tracker
    segment_tracker = SAMSegmentTracker(
        lines=lines,
        fps=fps,
        min_safe_time=args.min_safe_time,
        min_uncertain_time=args.min_uncertain_time,
        min_very_brief_time=args.min_very_brief_time
    )
    
    # Initialize frame overlay system
    frame_overlay = FrameOverlay(frame_height, frame_width, overlay_height=150)
    
    # Setup video writer with overlay
    if args.save_video:
        # Try different codecs for better compatibility
        try:
            # Use overlay dimensions
            output_width = frame_width
            output_height = frame_overlay.total_height
            
            # First try H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
            
            # Test if writer is opened
            if not out.isOpened():
                print("⚠️ H.264 codec failed, trying XVID...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_video_path = output_video_path.replace('.mp4', '.avi')  # Change extension for XVID
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
                
            if not out.isOpened():
                print("⚠️ XVID codec failed, trying MJPG...")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                output_video_path = output_video_path.replace('.avi', '.avi')  # Keep AVI for MJPG
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
                
            if not out.isOpened():
                print("❌ All codecs failed, disabling video output")
                args.save_video = False
            else:
                print(f"✅ Video writer initialized with codec: {fourcc}")
                print(f"📐 Output dimensions: {output_width}x{output_height}")
                
        except Exception as e:
            print(f"❌ Video writer setup failed: {e}")
            args.save_video = False
    
    # Processing variables
    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    detection_log = []
    
    print(f"\n🔥 Starting SAM processing...")
    print(f"   SAM Model: {args.sam_model}")
    print(f"   YOLO Model: {args.yolo_model}")
    print(f"   Device: {args.device}")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if specified
            if frame_idx % args.skip_frames != 0:
                frame_idx += 1
                continue
            
            # Max frames limit
            if args.max_frames and processed_frames >= args.max_frames:
                break
            
            # SAM detection and segmentation
            segmented_frame, detections = sam_logic.detect_and_segment(frame)
            
            # Debug output removed for cleaner execution
            
            # Update segment tracker
            crossings = segment_tracker.update(frame_idx, detections)
            
            # Debug crossings removed for cleaner execution
            
            # Log detections
            for detection in detections:
                log_entry = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'bbox_x1': detection['bbox'][0],
                    'bbox_y1': detection['bbox'][1], 
                    'bbox_x2': detection['bbox'][2],
                    'bbox_y2': detection['bbox'][3],
                    'mask_area': np.sum(detection['mask']) if detection['mask'] is not None else 0,
                    'mask_score': detection.get('mask_score', 0.0)
                }
                detection_log.append(log_entry)
            
            # Log crossings
            for crossing in crossings:
                crossing_entry = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'track_id': crossing['track_id'],
                    'class': crossing['class'],
                    'line_id': crossing['line_id'],
                    'direction': crossing['direction'],
                    'duration_frames': crossing['duration_frames'],
                    'duration_seconds': crossing['duration_seconds'],
                    'position_x': crossing['position'][0],
                    'position_y': crossing['position'][1]
                }
                segment_tracker.line_crossings.append(crossing_entry)
            
            # Draw lines on frame
            for i, line_point in enumerate(config.LINE_POINTS):
                cv2.line(segmented_frame, 
                        (line_point.x, 0), 
                        (line_point.x, frame_height),
                        (0, 255, 255), 2)  # Yellow lines
                cv2.putText(segmented_frame, f"L{i+1}", 
                           (line_point.x + 5, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add line crossing indicators to main frame
            segmented_frame = frame_overlay.draw_line_indicators(segmented_frame, crossings)
            
            # Create overlay with all information
            overlay_frame = frame_overlay.create_complete_overlay(
                original_frame=segmented_frame,
                line_crossings=crossings,
                detections=detections,
                current_frame=frame_idx,
                total_frames=total_frames,
                fps=fps,
                processing_time=None,
                timestamp=f"Frame {frame_idx}"
            )
            
            # Save frame with overlay
            if args.save_video:
                out.write(overlay_frame)
            
            # Show frame with overlay
            if args.show_video:
                cv2.imshow('SAM + LineLogic with Overlay', overlay_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if processed_frames % 100 == 0 and args.verbose:
                elapsed = time.time() - start_time
                fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                print(f"🟢 Processed frame {frame_idx} | Speed: {fps_processing:.1f} FPS")
            
            frame_idx += 1
            processed_frames += 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Processing error: {e}")
    
    finally:
        # Cleanup
        cap.release()
        if args.save_video:
            out.release()
        if args.show_video:
            cv2.destroyAllWindows()
    
    # Calculate final statistics
    processing_time = time.time() - start_time
    avg_fps = processed_frames / processing_time if processing_time > 0 else 0
    
    print(f"\n✅ Processing Complete!")
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
    print(f"\n{results_summary}")
    
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
            'sam_model': args.sam_model,
            'yolo_model': args.yolo_model,
            'confidence_threshold': args.confidence,
            'timestamp': timestamp
        })
    
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(results_csv_path, index=False)
    print(f"📊 Results saved: {results_csv_path}")
    
    if args.save_video:
        print(f"🎬 Processed video saved: {output_video_path}")

def main():
    """Main entry point."""
    args = setup_arguments()
    
    print("🎯 SAM + LineLogic Video Analysis")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"SAM Model: {args.sam_model}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    try:
        run_sam_analysis(args)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()