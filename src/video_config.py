"""
Video configuration for LineLogic.
Easily switch between different input videos.
"""

import os
from datetime import datetime

# Available videos in the videos directory
AVAILABLE_VIDEOS = {
    "1": {
        "name": "IMG_0015_blurred.MOV",
        "path": r"C:\Users\murat\Desktop\StajHW\LineLogic\videos\New Videos 2\IMG_0015_blurred.MOV",
        "description": "Test video 1"
    },
    "2": {
        "name": "MVI_6809_blurred.MOV", 
        "path": r"C:\Users\murat\Desktop\StajHW\LineLogic\videos\MVI_6809_blurred.MOV",
        "description": "Test video 2"
    },
    "3": {
        "name": "MVI_6817_blurred.MOV",
        "path": r"C:\Users\murat\Desktop\StajHW\LineLogic\videos\MVI_6817_blurred.MOV", 
        "description": "Test video 3"
    },
    "4": {
        "name": "6807-inital.MOV",
        "path": r"C:\Users\murat\Desktop\StajHW\LineLogic\videos\6807-inital.MOV",
        "description": "Initial test video"
    },
    "5": {
        "name": "IMG_0014_blurred.MOV",
        "path": r"C:\Users\murat\Desktop\StajHW\LineLogic\videos\New Videos 2\IMG_0014_blurred.MOV",
        "description": "Test video 5"
    },
    "6": {
        "name": "IMG_0014_blurred_1024x1024_right200.MOV",
        "path": r"C:\Users\murat\Desktop\StajHW\LineLogic\videos\New Videos 2\IMG_0014_blurred_1024x1024_right200.MOV",
        "description": "IMG_0014_blurred cropped 1024x1024, 200px right"
    }
}

def select_video():
    """
    Interactive video selection.
    
    Returns:
        str: Selected video path
    """
    print("📹 Available videos:")
    print("=" * 50)
    
    for key, video in AVAILABLE_VIDEOS.items():
        status = "✅" if os.path.exists(video["path"]) else "❌"
        print(f"{key}. {status} {video['name']} - {video['description']}")
    
    print("=" * 50)
    
    while True:
        choice = input("Select video (1-4) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            return None
        
        if choice in AVAILABLE_VIDEOS:
            video_path = AVAILABLE_VIDEOS[choice]["path"]
            if os.path.exists(video_path):
                print(f"✅ Selected: {AVAILABLE_VIDEOS[choice]['name']}")
                return video_path
            else:
                print(f"❌ Video not found: {video_path}")
        else:
            print("❌ Invalid choice. Please select 1-4 or 'q'.")

def get_video_by_name(video_name):
    """
    Get video path by name.
    
    Args:
        video_name: Name of the video file
    
    Returns:
        str: Video path or None if not found
    """
    for video in AVAILABLE_VIDEOS.values():
        if video["name"] == video_name:
            return video["path"] if os.path.exists(video["path"]) else None
    return None

def list_videos():
    """List all available videos with their status."""
    print("📹 Video Status:")
    print("=" * 50)
    
    for key, video in AVAILABLE_VIDEOS.items():
        status = "✅ Available" if os.path.exists(video["path"]) else "❌ Missing"
        print(f"{key}. {video['name']} - {status}")
        print(f"   Path: {video['path']}")
        print()

if __name__ == "__main__":
    # Test video selection
    list_videos()
    selected = select_video()
    if selected:
        print(f"🎯 Selected video: {selected}")
    else:
        print("👋 No video selected.") 