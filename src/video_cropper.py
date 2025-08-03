"""
Video cropping utilities for LineLogic
Handles video cropping with interactive preview and FFmpeg integration.
"""

import os
import sys
import subprocess
import cv2
import numpy as np

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)


def create_cropped_video(video_path, crop_params):
    """Create a cropped video using FFmpeg."""
    # Create cropped_videos directory if it doesn't exist
    videos_dir = os.path.dirname(video_path)
    cropped_dir = os.path.join(videos_dir, "cropped_videos")
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(cropped_dir, f"{base_name}_cropped.mp4")
    
    # FFmpeg path
    ffmpeg_path = os.path.join(project_root, "ffmpeg-7.1.1-essentials_build", "bin", "ffmpeg.exe")
    
    # Build FFmpeg command
    x, y, width, height = crop_params
    crop_filter = f"crop={width}:{height}:{x}:{y}"
    
    cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vf", crop_filter,
        "-c:a", "copy",  # Copy audio if present
        "-y",  # Overwrite output file
        output_path
    ]
    
    print(f"🎬 Creating cropped video...")
    print(f"   Crop: {width}x{height} at ({x},{y})")
    print(f"   Output: {output_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Cropped video created successfully!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating cropped video: {e}")
        print(f"FFmpeg error: {e.stderr}")
        return None


def interactive_crop_selection(video_path):
    """Interactive crop selection with video preview and visual markers."""
    while True:  # Loop for retry functionality
        print(f"\n🎬 Video cropping setup for: {os.path.basename(video_path)}")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video for preview")
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📐 Video dimensions: {width}x{height}")
        print(f"🎯 FPS: {fps:.2f}")
        print(f"📊 Total frames: {total_frames}")
        
        # Calculate center and default crop
        center_x = width // 2
        center_y = height // 2
        crop_width = min(1024, width)
        crop_height = min(1024, height)
        
        # Default crop from center
        default_x = max(0, center_x - crop_width // 2)
        default_y = max(0, center_y - crop_height // 2)
        
        print(f"\n📍 Default crop (from center):")
        print(f"   Position: ({default_x}, {default_y})")
        print(f"   Size: {crop_width}x{crop_height}")
        
        # Try to show video preview with markers
        print(f"\n🎥 Attempting to show video preview with crop markers...")
        
        # Check if GUI is available
        try:
            # Test if GUI works
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Test', test_frame)
            cv2.destroyAllWindows()
            gui_available = True
        except:
            gui_available = False
            print("⚠️  GUI preview not available - saving preview frames as images instead")
        
        if gui_available:
            print(f"   Press 'q' to quit preview, 's' to skip to crop setup")
            
            # Read a few frames to show preview
            frame_count = 0
            max_preview_frames = min(300, total_frames)  # Show first 10 seconds or 300 frames
            
            while frame_count < max_preview_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)  # Green center
                cv2.putText(frame, "CENTER", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw crop rectangle
                cv2.rectangle(frame, (default_x, default_y), (default_x + crop_width, default_y + crop_height), (0, 255, 255), 3)
                cv2.putText(frame, "CROP AREA", (default_x, default_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw 200-pixel markers from center
                marker_distance = 200
                for angle in [0, 90, 180, 270]:  # Right, Down, Left, Up
                    rad = angle * 3.14159 / 180
                    marker_x = int(center_x + marker_distance * np.cos(rad))
                    marker_y = int(center_y + marker_distance * np.sin(rad))
                    
                    # Only draw if within frame bounds
                    if 0 <= marker_x < width and 0 <= marker_y < height:
                        cv2.circle(frame, (marker_x, marker_y), 5, (255, 0, 0), -1)  # Blue markers
                        cv2.putText(frame, f"{marker_distance}px", (marker_x + 10, marker_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw frame info
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Press 'q' to quit, 's' to skip", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Video Preview - Crop Selection', frame)
                
                # Handle key presses
                key = cv2.waitKey(int(1000/fps)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    break
                    
                frame_count += 1
            
            cv2.destroyAllWindows()
        else:
            # Save preview frames as images
            preview_dir = os.path.join(os.path.dirname(video_path), "crop_preview")
            os.makedirs(preview_dir, exist_ok=True)
            
            frame_count = 0
            max_preview_frames = 1  # Save only 1 preview frame
            
            while frame_count < max_preview_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)  # Green center
                cv2.putText(frame, "CENTER", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw crop rectangle
                cv2.rectangle(frame, (default_x, default_y), (default_x + crop_width, default_y + crop_height), (0, 255, 255), 3)
                cv2.putText(frame, "CROP AREA", (default_x, default_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw 200-pixel markers from center
                marker_distance = 200
                for angle in [0, 90, 180, 270]:  # Right, Down, Left, Up
                    rad = angle * 3.14159 / 180
                    marker_x = int(center_x + marker_distance * np.cos(rad))
                    marker_y = int(center_y + marker_distance * np.sin(rad))
                    
                    # Only draw if within frame bounds
                    if 0 <= marker_x < width and 0 <= marker_y < height:
                        cv2.circle(frame, (marker_x, marker_y), 5, (255, 0, 0), -1)  # Blue markers
                        cv2.putText(frame, f"{marker_distance}px", (marker_x + 10, marker_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Save frame as image
                preview_path = os.path.join(preview_dir, f"preview_frame_{frame_count:03d}.jpg")
                cv2.imwrite(preview_path, frame)
                print(f"   Saved preview frame {frame_count + 1}/{max_preview_frames}: {preview_path}")
                
                frame_count += 1
            
            print(f"\n📸 Preview frames saved to: {preview_dir}")
            print(f"   Open these images to see the crop area and markers")
        
        cap.release()
        
        # Ask if user wants to modify crop
        modify = input("\nDo you want to modify the crop? (y/N): ").strip().lower()
        
        if modify == 'y':
            print("\n🎯 Crop modification:")
            print("   Enter new values (press Enter to keep default)")
            print("   Tip: Use the preview to determine good coordinates")
            
            try:
                new_x = input(f"   X position (default: {default_x}): ").strip()
                new_y = input(f"   Y position (default: {default_y}): ").strip()
                new_width = input(f"   Width (default: {crop_width}): ").strip()
                new_height = input(f"   Height (default: {crop_height}): ").strip()
                
                x = int(new_x) if new_x else default_x
                y = int(new_y) if new_y else default_y
                w = int(new_width) if new_width else crop_width
                h = int(new_height) if new_height else crop_height
                
                # Validate bounds
                x = max(0, min(x, width - w))
                y = max(0, min(y, height - h))
                w = min(w, width - x)
                h = min(h, height - y)
                
            except ValueError:
                print("❌ Invalid input, using default values")
                x, y, w, h = default_x, default_y, crop_width, crop_height
        else:
            x, y, w, h = default_x, default_y, crop_width, crop_height
        
        print(f"\n✅ Final crop parameters:")
        print(f"   Position: ({x}, {y})")
        print(f"   Size: {w}x{h}")
        
        # Create cropped video
        cropped_path = create_cropped_video(video_path, (x, y, w, h))
        if not cropped_path:
            print("❌ Failed to create cropped video")
            return None
        
        # Show preview of cropped result
        print(f"\n🎬 Showing preview of cropped video...")
        
        # Check if GUI is available for cropped preview
        try:
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Test', test_frame)
            cv2.destroyAllWindows()
            gui_available = True
        except:
            gui_available = False
        
        if gui_available:
            print(f"   Press any key to close preview")
            
            # Open cropped video and show first few frames
            cropped_cap = cv2.VideoCapture(cropped_path)
            if cropped_cap.isOpened():
                frame_count = 0
                max_preview_frames = min(100, int(cropped_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                
                while frame_count < max_preview_frames:
                    ret, frame = cropped_cap.read()
                    if not ret:
                        break
                    
                    # Add info text
                    cv2.putText(frame, f"CROPPED PREVIEW - Frame {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Press any key to close", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Cropped Video Preview', frame)
                    
                    # Wait for key press or auto-advance
                    key = cv2.waitKey(int(1000/fps)) & 0xFF
                    if key != 255:  # Any key pressed
                        break
                        
                    frame_count += 1
                
                cropped_cap.release()
                cv2.destroyAllWindows()
        else:
            # Save cropped preview frames as images
            cropped_preview_dir = os.path.join(os.path.dirname(cropped_path), "cropped_preview")
            os.makedirs(cropped_preview_dir, exist_ok=True)
            
            cropped_cap = cv2.VideoCapture(cropped_path)
            if cropped_cap.isOpened():
                frame_count = 0
                max_preview_frames = 1  # Save only 1 preview frame
                
                while frame_count < max_preview_frames:
                    ret, frame = cropped_cap.read()
                    if not ret:
                        break
                    
                    # Add info text
                    cv2.putText(frame, f"CROPPED PREVIEW - Frame {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame as image
                    preview_path = os.path.join(cropped_preview_dir, f"cropped_preview_{frame_count:03d}.jpg")
                    cv2.imwrite(preview_path, frame)
                    print(f"   Saved cropped preview frame {frame_count + 1}/{max_preview_frames}: {preview_path}")
                    
                    frame_count += 1
                
                cropped_cap.release()
                print(f"\n📸 Cropped preview frames saved to: {cropped_preview_dir}")
                print(f"   Open these images to see the cropped result")
        
        # Ask user if they're satisfied with the result
        while True:
            satisfied = input(f"\nAre you satisfied with the cropped result? (y/N): ").strip().lower()
            if satisfied == 'y':
                print("✅ Cropping completed successfully!")
                return cropped_path
            elif satisfied == 'n':
                print("🔄 Starting over with new crop parameters...")
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no")
        
        # If we get here, user wants to retry, so continue the loop 