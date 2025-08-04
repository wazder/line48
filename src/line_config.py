"""
Line configuration utilities for LineLogic
Handles interactive line placement setup with preview functionality.
"""

import os
import sys
import cv2
import numpy as np

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

# Line configuration defaults
LINE_HEIGHT = 1080
LINE_POINTS = [
    (880, 0),   # Line 1
    (960, 0),   # Line 2
    (1040, 0),  # Line 3
    (1120, 0)   # Line 4
]

def calculate_line_positions(base_x, line_spacing, num_lines=4):
    """Calculate X positions for all lines."""
    return [base_x + i * line_spacing for i in range(num_lines)]


def create_line_preview(video_path, base_x, line_spacing, line_height, num_lines=4):
    """Create a preview image showing line placements."""
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video for line preview")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read video frame for line preview")
        cap.release()
        return None
    
    cap.release()
    
    # Calculate line positions
    line_positions = calculate_line_positions(base_x, line_spacing, num_lines)
    
    # Draw lines on frame
    for i, x_pos in enumerate(line_positions):
        # Draw line
        cv2.line(frame, (x_pos, 0), (x_pos, height), (0, 255, 0), 3)
        
        # Add line number and position
        cv2.putText(frame, f"Line {i+1}: x={x_pos}", (x_pos + 10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw line height marker
        cv2.line(frame, (x_pos - 20, line_height), (x_pos + 20, line_height), (255, 0, 0), 2)
        cv2.putText(frame, f"y={line_height}", (x_pos + 10, line_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Add title and info
    cv2.putText(frame, "LINE PLACEMENT PREVIEW", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Base X: {base_x}, Spacing: {line_spacing}, Height: {line_height}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Line positions: {line_positions}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def interactive_line_setup(video_path):
    """Interactive line placement setup with preview."""
    print(f"\n📏 Line Placement Setup")
    print("-" * 30)
    print("Configure where the 4 detection lines will be placed:")
    print("• Base X: Starting X position of the first line")
    print("• Line Spacing: Distance between consecutive lines")
    print("• Line Height: Y position where all lines will be placed")
    
    # Default values (from config.py)
    default_base_x = 880
    default_line_spacing = 80
    default_line_height = 1080
    
    while True:  # Loop for retry functionality
        print(f"\n📍 Default line parameters:")
        print(f"   Base X: {default_base_x}")
        print(f"   Line Spacing: {default_line_spacing}")
        print(f"   Line Height: {default_line_height}")
        
        # Calculate default line positions
        default_positions = calculate_line_positions(default_base_x, default_line_spacing)
        print(f"   Line positions: {default_positions}")
        
        # Ask if user wants to modify
        modify = input("\nDo you want to modify the line parameters? (y/N): ").strip().lower()
        
        if modify == 'y':
            print("\n🎯 Line parameter modification:")
            print("   Enter new values (press Enter to keep default)")
            
            try:
                new_base_x = input(f"   Base X (default: {default_base_x}): ").strip()
                new_spacing = input(f"   Line Spacing (default: {default_line_spacing}): ").strip()
                new_height = input(f"   Line Height (default: {default_line_height}): ").strip()
                
                base_x = int(new_base_x) if new_base_x else default_base_x
                line_spacing = int(new_spacing) if new_spacing else default_line_spacing
                line_height = int(new_height) if new_height else default_line_height
                
                # Validate values
                if base_x < 0 or line_spacing < 0 or line_height < 0:
                    print("❌ Values must be positive")
                    continue
                
            except ValueError:
                print("❌ Invalid input, using default values")
                base_x, line_spacing, line_height = default_base_x, default_line_spacing, default_line_height
        else:
            base_x, line_spacing, line_height = default_base_x, default_line_spacing, default_line_height
        
        # Calculate final line positions
        line_positions = calculate_line_positions(base_x, line_spacing)
        
        print(f"\n✅ Final line parameters:")
        print(f"   Base X: {base_x}")
        print(f"   Line Spacing: {line_spacing}")
        print(f"   Line Height: {line_height}")
        print(f"   Line positions: {line_positions}")
        
        # Create and save line preview
        print(f"\n📸 Creating line placement preview...")
        
        preview_frame = create_line_preview(video_path, base_x, line_spacing, line_height)
        if preview_frame is not None:
            # Save preview image
            preview_dir = os.path.join(os.path.dirname(video_path), "line_preview")
            os.makedirs(preview_dir, exist_ok=True)
            
            preview_path = os.path.join(preview_dir, "line_placement_preview.jpg")
            cv2.imwrite(preview_path, preview_frame)
            print(f"   Line preview saved to: {preview_path}")
            print(f"   Open this image to see where the lines will be placed")
        
        # Ask user if they're satisfied with the result
        while True:
            satisfied = input(f"\nAre you satisfied with the line placement? (y/N): ").strip().lower()
            if satisfied == 'y':
                print("✅ Line placement configured successfully!")
                return {
                    'base_x': base_x,
                    'line_spacing': line_spacing,
                    'line_height': line_height,
                    'line_positions': line_positions
                }
            elif satisfied == 'n':
                print("🔄 Starting over with new line parameters...")
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no")
        
        # If we get here, user wants to retry, so continue the loop 