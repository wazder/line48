#!/usr/bin/env python3
"""Test script for frame overlay system"""

import cv2
import numpy as np
from src.frame_overlay import FrameOverlay

def test_overlay():
    print("🎨 Frame Overlay Test")
    print("=" * 30)
    
    # Create test frame
    frame_width = 640
    frame_height = 480
    test_frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
    
    # Add some test content to frame
    cv2.putText(test_frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(test_frame, (100, 100), (300, 200), (0, 255, 0), 2)
    
    # Initialize overlay
    overlay = FrameOverlay(frame_height, frame_width, overlay_height=150)
    
    # Create test line crossings
    test_crossings = [
        {
            'track_id': 1,
            'class': 'person',
            'line_id': 1,
            'direction': 'IN',
            'position': (200, 150),
            'confidence': 0.85
        },
        {
            'track_id': 2,
            'class': 'backpack',
            'line_id': 2,
            'direction': 'OUT',
            'position': (400, 200),
            'confidence': 0.92
        },
        {
            'track_id': 3,
            'class': 'handbag',
            'line_id': 1,
            'direction': 'IN',
            'position': (300, 180),
            'confidence': 0.78
        }
    ]
    
    # Create test detections
    test_detections = [
        {'class': 'person', 'confidence': 0.85, 'track_id': 1},
        {'class': 'backpack', 'confidence': 0.92, 'track_id': 2},
        {'class': 'handbag', 'confidence': 0.78, 'track_id': 3},
        {'class': 'suitcase', 'confidence': 0.65, 'track_id': 4}
    ]
    
    # Create overlay frame
    overlay_frame = overlay.create_complete_overlay(
        original_frame=test_frame,
        line_crossings=test_crossings,
        detections=test_detections,
        current_frame=100,
        total_frames=500,
        fps=30.0,
        processing_time=0.033,
        timestamp="Test Frame 100"
    )
    
    # Save test result
    output_path = "test_overlay_output.jpg"
    cv2.imwrite(output_path, overlay_frame)
    
    print(f"✅ Overlay test completed!")
    print(f"📐 Input frame: {frame_width}x{frame_height}")
    print(f"📐 Output frame: {frame_width}x{overlay.total_height}")
    print(f"💾 Saved to: {output_path}")
    
    # Display frame info
    print(f"\n📊 Test Data:")
    print(f"   Line crossings: {len(test_crossings)}")
    print(f"   Detections: {len(test_detections)}")
    
    return overlay_frame

if __name__ == "__main__":
    test_overlay() 