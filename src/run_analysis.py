"""
Enhanced LineLogic Analysis Runner with Command-Line Interface
Main entry point for video analysis with modular architecture.
"""

import os
import argparse
import csv
from datetime import datetime
import supervision as sv
from supervision import VideoInfo
import numpy as np
import cv2
from utils import load_model
from frame_logic import FrameBasedTracker
from video_config import select_video, list_videos, get_video_by_name
# Import modular components
from video_utils import interactive_video_selection, display_video_list
from video_cropper import interactive_crop_selection
from line_config import interactive_line_setup
from results_exporter import export_results_csv
from frame_overlay import FrameOverlay
# from detailed_video_processor import process_video_with_detailed_info
# These imports will be moved to where they're needed to avoid circular imports

def run_analysis(source_video_path, use_frame_logic=True, line_config=None, **kwargs):
    """
    Run analysis on a video with configurable parameters.
    
    Args:
        source_video_path: Path to source video
        use_frame_logic: Whether to use frame-based validation
        line_config: Line configuration (optional)
        **kwargs: Additional parameters (confidence, iou, imgsz, detailed_info, etc.)
    """
    
    # Import config after setting up video path
    import config
    
    # Override source video path
    config.SOURCE_VIDEO_PATH = source_video_path
    
    # Regenerate output paths based on new source video
    source_video_name = os.path.splitext(os.path.basename(source_video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = f"outputs/{source_video_name}_processed_{timestamp}"
    base_log_name = f"logs/{source_video_name}_log_{timestamp}"
    base_results_name = f"logs/{source_video_name}_results_{timestamp}"
    
    config.TARGET_VIDEO_PATH = config.get_next_filename(base_output_name, ".mp4")
    config.LOG_CSV_PATH = config.get_next_filename(base_log_name, ".csv")
    RESULTS_CSV_PATH = config.get_next_filename(base_results_name, ".csv")
    
    # Update video dimensions for overlay
    video_info = VideoInfo.from_video_path(source_video_path)
    overlay_height = 150
    total_height = video_info.height + overlay_height
    
    # Create output directories
    os.makedirs(os.path.dirname(config.TARGET_VIDEO_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    
    print(f"🎬 Processing: {os.path.basename(source_video_path)}")
    print(f"📁 Output: {config.TARGET_VIDEO_PATH}")
    print(f"📝 Log: {config.LOG_CSV_PATH}")
    print(f"📊 Results: {RESULTS_CSV_PATH}")
    
    # Load model
    model_single = load_model()
    
    # Apply configurable parameters
    print("🔧 Model parameters:")
    model_single.conf = kwargs.get('confidence', 0.25)
    model_single.iou = kwargs.get('iou', 0.45)
    model_single.imgsz = kwargs.get('imgsz', 1024)
    
    print(f"   Confidence threshold: {model_single.conf}")
    print(f"   NMS threshold: {model_single.iou}")
    print(f"   Image size: {model_single.imgsz}")
    
    # Load video info
    video_info = VideoInfo.from_video_path(source_video_path)
    print(f"📦 Total frames: {video_info.total_frames}")
    print(f"🎯 FPS: {video_info.fps}")
    
    # Store parameters for results CSV
    parameters = {
        'video_path': source_video_path,
        'video_name': os.path.basename(source_video_path),
        'total_frames': video_info.total_frames,
        'fps': video_info.fps,
        'confidence_threshold': model_single.conf,
        'iou_threshold': model_single.iou,
        'image_size': model_single.imgsz,
        'use_frame_logic': use_frame_logic,
        'max_frames': kwargs.get('max_frames'),
        'timestamp': timestamp
    }
    
    # Add line configuration to parameters
    if line_config:
        parameters.update({
            'line_base_x': line_config['base_x'],
            'line_spacing': line_config['line_spacing'],
            'line_height': line_config['line_height'],
            'line_positions': str(line_config['line_positions'])
        })
    
    if use_frame_logic:
        # Frame-based logic parameters
        min_safe_time = kwargs.get('min_safe_time', 0.5)
        min_uncertain_time = kwargs.get('min_uncertain_time', 0.28)
        min_very_brief_time = kwargs.get('min_very_brief_time', 0.17)
        
        parameters.update({
            'min_safe_time': min_safe_time,
            'min_uncertain_time': min_uncertain_time,
            'min_very_brief_time': min_very_brief_time
        })
        
        print(f"   Min safe time: {min_safe_time}s")
        print(f"   Min uncertain time: {min_uncertain_time}s")
        print(f"   Min very brief time: {min_very_brief_time}s")
        
        # Initialize frame-based tracker
        tracker = FrameBasedTracker(
            fps=video_info.fps,
            min_safe_time=min_safe_time,
            min_uncertain_time=min_uncertain_time,
            min_very_brief_time=min_very_brief_time
        )
        
        # Setup lines and annotators
        from config import LINE_IDS
        
        # Use custom line configuration if provided, otherwise use defaults
        if line_config:
            line_positions = line_config['line_positions']
            line_height = line_config['line_height']
            LINE_POINTS = [sv.Point(x, 0) for x in line_positions]
            LINES = [sv.LineZone(start=p, end=sv.Point(p.x, line_height)) for p in LINE_POINTS]
        else:
            from config import LINE_POINTS, LINE_HEIGHT
            LINES = [sv.LineZone(start=p, end=sv.Point(p.x, LINE_HEIGHT)) for p in LINE_POINTS]
        line_annotators = [
            sv.LineZoneAnnotator(
                display_in_count=False,
                display_out_count=False,
                text_thickness=2,
                text_scale=1.0
            )
            for _ in LINE_IDS
        ]
        
                # Initialize frame overlay system
        frame_overlay = FrameOverlay(video_info.height, video_info.width, overlay_height=150)
        
        # Create callback with frame logic and overlay
        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            # Check max frames limit
            max_frames = kwargs.get('max_frames')
            if max_frames is not None and index >= max_frames:
                raise StopIteration
            
            # Run inference
            results = model_single(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter to target classes
            COCO_NAMES = model_single.model.names
            SELECTED_CLASSES = ["person", "backpack", "handbag", "suitcase"]
            SELECTED_CLASS_IDS = [k for k, v in COCO_NAMES.items() if v in SELECTED_CLASSES]
            detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
            
            # Track objects
            detections = tracker.byte_tracker.update_with_detections(detections)
            
            # Update object presence
            tracker.update_object_presence(detections, index, COCO_NAMES)
            
            # Process line crossings
            crossings = tracker.process_line_crossing(detections, index, LINES, LINE_IDS, COCO_NAMES)
            
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
                current_frame=index,
                total_frames=video_info.total_frames,
                fps=video_info.fps,
                processing_time=None,
                timestamp=f"Frame {index}"
            )
            
            if index % 100 == 0:
                print(f"🟢 Processed frame {index}")
 
            return overlay_frame
        
        # Process video
        detailed_info = kwargs.get('detailed_info', False)
        
        if detailed_info:
            print("🔍 Detailed processing not available, using standard processing...")
            sv.process_video(
                source_path=source_video_path,
                target_path=config.TARGET_VIDEO_PATH,
                callback=callback
            )
        else:
            try:
                sv.process_video(
                    source_path=source_video_path,
                    target_path=config.TARGET_VIDEO_PATH,
                    callback=callback
                )
            except StopIteration:
                print("\n🛑 Stopped manually.")
        
        # Get results
        results_summary = tracker.get_results_summary()
        
        # Print results
        print("\n📊 Frame-based tracking results:")
        for class_name, counts in results_summary.items():
            safe = counts["safe"]
            uncertain = counts["uncertain"]
            total = counts["total"]
            print(f"{class_name:<10} → Safe: {safe}, Uncertain: {uncertain}, Total: {total}")
        
        # Export enhanced CSV log
        with open(config.LOG_CSV_PATH, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Object ID", "Class", "Line Number", "Direction", "Frame", 
                "Timestamp (min:sec)", "Confidence", "Tracking Duration"
            ])
            writer.writerows(tracker.get_log_rows())
        
        print(f"\n📁 Enhanced log saved to {config.LOG_CSV_PATH}")
        print(f"🎬 Output video saved to {config.TARGET_VIDEO_PATH}")
        
        # Print discarded crossings summary
        discarded = tracker.get_discarded_summary()
        print(f"\n🗑️ Discarded crossings (too brief to count): {len(discarded)}")
        if discarded:
            print("Sample discarded crossings (tid, class, line, dir, frame, duration_frames):")
            for row in discarded[:10]:
                print(row)
        
    else:
        # Use basic logic
        from logic import create_callback
        
        # Create callback with basic logic
        max_frames = kwargs.get('max_frames')
        callback = create_callback(model_single, video_info, MAX_FRAMES=max_frames, line_config=line_config)
        
        # Process video
        try:
            sv.process_video(
                source_path=source_video_path,
                target_path=config.TARGET_VIDEO_PATH,
                callback=callback
            )
        except StopIteration:
            print("\n🛑 Stopped manually.")
        
        # Get results
        results_summary = {}
        for class_name, counts in callback.per_class_counter.items():
            results_summary[class_name] = {
                "in": counts['in'],
                "out": counts['out'],
                "total": counts['in'] + counts['out']
            }
        
        # Print results
        print("\n📊 Basic tracking results:")
        for class_name, counts in results_summary.items():
            print(f"{class_name:<10} → IN: {counts['in']}, OUT: {counts['out']}")
        
        # Export CSV log
        with open(config.LOG_CSV_PATH, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Object ID", "Class", "Line Number", "Direction", "Frame", "Timestamp (min:sec)"])
            writer.writerows(callback.log_rows)
        
        print(f"\n📁 Log saved to {config.LOG_CSV_PATH}")
        print(f"🎬 Output video saved to {config.TARGET_VIDEO_PATH}")
    
    # Export comprehensive results CSV
    export_results_csv(RESULTS_CSV_PATH, parameters, results_summary)
    print(f"📊 Comprehensive results saved to {RESULTS_CSV_PATH}")



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LineLogic Enhanced Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (includes cropping option)
  python run_analysis.py
  
  # Process specific video with frame logic
  python run_analysis.py --video "videos/IMG_0015_blurred.MOV" --frame-logic
  
  # Process with custom parameters
  python run_analysis.py --video "videos/IMG_0015_blurred.MOV" --confidence 0.3 --iou 0.4 --imgsz 1280
  
  # Process with cropping
  python run_analysis.py --video "videos/IMG_0015_blurred.MOV" --crop
  
  # Process limited frames
  python run_analysis.py --video "videos/IMG_0015_blurred.MOV" --max-frames 100
  
  # List available videos
  python run_analysis.py --list-videos
  
  # Create cropped video only (no processing)
  python run_analysis.py --crop-video "videos/IMG_0015_blurred.MOV"
        """
    )
    
    # Video selection
    parser.add_argument("--video", "-v", help="Path to video file to process")
    parser.add_argument("--list-videos", "-l", action="store_true", help="List all available videos")
    parser.add_argument("--crop-video", "-c", help="Path to video file to crop")
    parser.add_argument("--crop", action="store_true", help="Crop the selected video before processing")
    
    # Logic selection
    parser.add_argument("--frame-logic", "-f", action="store_true", help="Use frame-based logic (default)")
    parser.add_argument("--basic-logic", "-b", action="store_true", help="Use basic logic")
    
    # Model parameters
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size (default: 1024)")
    
    # Frame logic parameters
    parser.add_argument("--min-safe-time", type=float, default=0.5, help="Min safe time in seconds (default: 0.5)")
    parser.add_argument("--min-uncertain-time", type=float, default=0.28, help="Min uncertain time in seconds (default: 0.28)")
    parser.add_argument("--min-very-brief-time", type=float, default=0.17, help="Min very brief time in seconds (default: 0.17)")
    
    # Processing parameters
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (default: None = all frames)")
    parser.add_argument("--detailed-info", action="store_true", 
                       help="Show detailed frame-by-frame information overlay")
    
    return parser.parse_args()

def interactive_setup():
    """Interactive setup with all options and defaults."""
    print("\n🔧 Interactive Setup - All Options")
    print("=" * 50)
    
    # Step 1: Video Selection
    print("\n📹 Step 1: Video Selection")
    print("-" * 30)
    source_video = interactive_video_selection()
    if not source_video:
        print("👋 No video selected. Exiting.")
        return None, None
    
    # Step 1.5: Video Cropping (Optional)
    print("\n🎬 Step 1.5: Video Cropping (Optional)")
    print("-" * 30)
    print("Do you want to crop the video before processing?")
    print("• Cropping can help focus on specific areas")
    print("• Shows video preview with center point and crop markers")
    print("• Creates cropped video in videos/cropped_videos/ directory")
    
    while True:
        crop_choice = input("\nCrop video? (y/N) or press Enter for default (No): ").strip().lower()
        if not crop_choice or crop_choice == 'n':
            print("✅ Skipping video cropping")
            break
        elif crop_choice == 'y':
            print("🎬 Starting video cropping process...")
            cropped_video = interactive_crop_selection(source_video)
            if cropped_video:
                print(f"✅ Cropped video created: {os.path.basename(cropped_video)}")
                source_video = cropped_video
                break
            else:
                print("❌ Cropping failed, using original video")
                break
        else:
            print("❌ Please enter 'y' for yes, 'n' for no, or press Enter for default")
    
    # Step 1.6: Line Placement Configuration
    print("\n📏 Step 1.6: Line Placement Configuration")
    print("-" * 30)
    print("Configure where the detection lines will be placed:")
    print("• Shows preview of line positions on the video")
    print("• Allows customization of line spacing and position")
    
    line_config = interactive_line_setup(source_video)
    if not line_config:
        print("❌ Line configuration failed, using defaults")
        line_config = {
            'base_x': 880,
            'line_spacing': 80,
            'line_height': 1080,
            'line_positions': [880, 960, 1040, 1120]
        }
    
    # Step 2: Logic Selection
    print("\n🔧 Step 2: Logic Selection")
    print("-" * 30)
    print("Available options:")
    print("1. Frame-based logic (recommended) - Filters brief detections")
    print("2. Basic logic - Simple tracking without filtering")
    
    while True:
        choice = input("\nSelect logic (1/2) or press Enter for default (1): ").strip()
        if not choice:
            use_frame_logic = True
            print("✅ Using default: Frame-based logic")
            break
        elif choice == "1":
            use_frame_logic = True
            print("✅ Selected: Frame-based logic")
            break
        elif choice == "2":
            use_frame_logic = False
            print("✅ Selected: Basic logic")
            break
        else:
            print("❌ Please enter 1, 2, or press Enter for default")
    
    # Step 3: Model Parameters
    print("\n🧠 Step 3: Model Parameters")
    print("-" * 30)
    print("Available parameters with current defaults:")
    print("• Confidence threshold: 0.25 (0.0-1.0, higher = more confident)")
    print("• IoU threshold: 0.45 (0.0-1.0, lower = more detections)")
    print("• Image size: 1024 (pixels, larger = better for small objects)")
    
    # Confidence
    while True:
        conf_input = input(f"\nConfidence threshold (0.0-1.0) or press Enter for default (0.25): ").strip()
        if not conf_input:
            confidence = 0.25
            print("✅ Using default confidence: 0.25")
            break
        try:
            confidence = float(conf_input)
            if 0.0 <= confidence <= 1.0:
                print(f"✅ Confidence set to: {confidence}")
                break
            else:
                print("❌ Confidence must be between 0.0 and 1.0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # IoU
    while True:
        iou_input = input(f"IoU threshold (0.0-1.0) or press Enter for default (0.45): ").strip()
        if not iou_input:
            iou = 0.45
            print("✅ Using default IoU: 0.45")
            break
        try:
            iou = float(iou_input)
            if 0.0 <= iou <= 1.0:
                print(f"✅ IoU set to: {iou}")
                break
            else:
                print("❌ IoU must be between 0.0 and 1.0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Image size
    while True:
        imgsz_input = input(f"Image size (pixels) or press Enter for default (1024): ").strip()
        if not imgsz_input:
            imgsz = 1024
            print("✅ Using default image size: 1024")
            break
        try:
            imgsz = int(imgsz_input)
            if imgsz > 0:
                print(f"✅ Image size set to: {imgsz}")
                break
            else:
                print("❌ Image size must be positive")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Step 4: Frame Logic Parameters (only if using frame logic)
    min_safe_time = 0.5
    min_uncertain_time = 0.28
    min_very_brief_time = 0.17
    
    if use_frame_logic:
        print("\n⏱️ Step 4: Frame Logic Parameters")
        print("-" * 30)
        print("Time thresholds for frame-based validation:")
        print("• Min safe time: 0.5s (minimum time for 'safe' predictions)")
        print("• Min uncertain time: 0.28s (minimum time for 'uncertain' predictions)")
        print("• Min very brief time: 0.17s (minimum time for 'very brief' predictions)")
        print("Note: Lower times = more detections, but may include false positives")
        
        # Min safe time
        while True:
            safe_input = input(f"\nMin safe time (seconds) or press Enter for default (0.5): ").strip()
            if not safe_input:
                min_safe_time = 0.5
                print("✅ Using default min safe time: 0.5s")
                break
            try:
                min_safe_time = float(safe_input)
                if min_safe_time > 0:
                    print(f"✅ Min safe time set to: {min_safe_time}s")
                    break
                else:
                    print("❌ Time must be positive")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Min uncertain time
        while True:
            uncertain_input = input(f"Min uncertain time (seconds) or press Enter for default (0.28): ").strip()
            if not uncertain_input:
                min_uncertain_time = 0.28
                print("✅ Using default min uncertain time: 0.28s")
                break
            try:
                min_uncertain_time = float(uncertain_input)
                if min_uncertain_time > 0:
                    print(f"✅ Min uncertain time set to: {min_uncertain_time}s")
                    break
                else:
                    print("❌ Time must be positive")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Min very brief time
        while True:
            brief_input = input(f"Min very brief time (seconds) or press Enter for default (0.17): ").strip()
            if not brief_input:
                min_very_brief_time = 0.17
                print("✅ Using default min very brief time: 0.17s")
                break
            try:
                min_very_brief_time = float(brief_input)
                if min_very_brief_time > 0:
                    print(f"✅ Min very brief time set to: {min_very_brief_time}s")
                    break
                else:
                    print("❌ Time must be positive")
            except ValueError:
                print("❌ Please enter a valid number")
    
    # Step 5: Max Frames (Optional)
    print("\n🎬 Step 5: Max Frames (Optional)")
    print("-" * 30)
    print("Limit the number of frames to process:")
    print("• None = Process entire video")
    print("• Enter a number = Process only that many frames")
    print("• Useful for testing or processing specific parts")
    
    while True:
        max_frames_input = input(f"\nMax frames to process or press Enter for default (None = all frames): ").strip()
        if not max_frames_input:
            max_frames = None
            print("✅ Using default: Process all frames")
            break
        try:
            max_frames = int(max_frames_input)
            if max_frames > 0:
                print(f"✅ Max frames set to: {max_frames}")
                break
            else:
                print("❌ Max frames must be positive")
        except ValueError:
            print("❌ Please enter a valid number")
    
        # Prepare parameters
    params = {
         'confidence': confidence,
         'iou': iou,
         'imgsz': imgsz,
         'min_safe_time': min_safe_time,
         'min_uncertain_time': min_uncertain_time,
         'min_very_brief_time': min_very_brief_time,
         'max_frames': max_frames
     }
    
    # Summary
    print("\n📋 Configuration Summary")
    print("-" * 30)
    print(f"Video: {os.path.basename(source_video)}")
    if "cropped" in source_video.lower():
        print("   (Cropped video)")
    print(f"Logic: {'Frame-based' if use_frame_logic else 'Basic'}")
    print(f"Confidence: {confidence}")
    print(f"IoU: {iou}")
    print(f"Image size: {imgsz}")
    if use_frame_logic:
        print(f"Min safe time: {min_safe_time}s")
        print(f"Min uncertain time: {min_uncertain_time}s")
        print(f"Min very brief time: {min_very_brief_time}s")
    print(f"Max frames: {max_frames if max_frames else 'All frames'}")
    print(f"Line Base X: {line_config['base_x']}")
    print(f"Line Spacing: {line_config['line_spacing']}")
    print(f"Line Height: {line_config['line_height']}")
    print(f"Line Positions: {line_config['line_positions']}")
    
    return source_video, use_frame_logic, params, line_config

def main():
    """Main function with command-line interface."""
    args = parse_arguments()
    
    print("🚀 LineLogic Enhanced Analysis Runner")
    print("=" * 50)
    
    # Handle list videos
    if args.list_videos:
        display_video_list()
        return
    
    # Handle crop video
    if args.crop_video:
        if not os.path.exists(args.crop_video):
            print(f"❌ Video not found: {args.crop_video}")
            return
        
        cropped_path = interactive_crop_selection(args.crop_video)
        if cropped_path:
            print(f"\n✅ Cropped video ready: {cropped_path}")
            process_cropped = input("Process the cropped video now? (y/N): ").strip().lower()
            if process_cropped == 'y':
                args.video = cropped_path
            else:
                return
        else:
            return
    
    # Check if any command-line arguments were provided
    if any([args.video, args.frame_logic, args.basic_logic, args.crop, args.max_frames,
            args.confidence != 0.25, args.iou != 0.45, args.imgsz != 1024,
            args.min_safe_time != 0.5, args.min_uncertain_time != 0.28, args.min_very_brief_time != 0.17]):
        # Use command-line arguments
        if args.video:
            source_video = args.video
            if not os.path.exists(source_video):
                print(f"❌ Video not found: {source_video}")
                return
        else:
            source_video = interactive_video_selection()
            if not source_video:
                print("👋 No video selected. Exiting.")
                return
        
        # Handle cropping if requested
        if args.crop:
            print("🎬 Cropping video as requested...")
            cropped_video = interactive_crop_selection(source_video)
            if cropped_video:
                print(f"✅ Using cropped video: {os.path.basename(cropped_video)}")
                source_video = cropped_video
            else:
                print("❌ Cropping failed, using original video")
        
        # Determine logic type
        if args.basic_logic:
            use_frame_logic = False
            print("🔧 Using basic logic")
        else:
            use_frame_logic = True
            print("🎯 Using frame-based logic")
        
        # Prepare parameters
        params = {
            'confidence': args.confidence,
            'iou': args.iou,
            'imgsz': args.imgsz,
            'min_safe_time': args.min_safe_time,
            'min_uncertain_time': args.min_uncertain_time,
            'min_very_brief_time': args.min_very_brief_time,
            'max_frames': args.max_frames,
            'detailed_info': args.detailed_info
        }
        line_config = None
    else:
        # Use interactive setup
        result = interactive_setup()
        if result[0] is None:
            return
        source_video, use_frame_logic, params, line_config = result
    
    # Run analysis
    print("\n" + "="*50)
    run_analysis(source_video, use_frame_logic, line_config=line_config, **params)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 