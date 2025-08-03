"""
Video utilities for LineLogic
Handles video discovery, listing, and selection functionality.
"""

import os
import sys

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)


def find_videos_in_directory(directory):
    """Find all video files in a directory recursively."""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.MOV', '.MP4'}
    videos = []
    
    if not os.path.exists(directory):
        return videos
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Skip macOS metadata files and very small files
            if file.startswith('._') or file.startswith('.DS_Store'):
                continue
                
            if any(file.lower().endswith(ext.lower()) for ext in video_extensions):
                full_path = os.path.join(root, file)
                file_size = os.path.getsize(full_path)
                
                # Skip files smaller than 1MB (likely not real videos)
                if file_size < 1024 * 1024:
                    continue
                    
                relative_path = os.path.relpath(full_path, directory)
                videos.append((relative_path, full_path))
    
    return sorted(videos)


def list_all_videos():
    """List all videos in the videos directory and subdirectories."""
    videos_dir = os.path.join(project_root, "videos")
    all_videos = []
    seen_paths = set()
    
    # Get videos from main videos directory
    main_videos = find_videos_in_directory(videos_dir)
    for relative_path, full_path in main_videos:
        if full_path not in seen_paths:
            all_videos.append((relative_path, full_path))
            seen_paths.add(full_path)
    
    # Get videos from New Videos 2 subdirectory
    new_videos_dir = os.path.join(videos_dir, "New Videos 2")
    new_videos = find_videos_in_directory(new_videos_dir)
    for relative_path, full_path in new_videos:
        if full_path not in seen_paths:
            all_videos.append((relative_path, full_path))
            seen_paths.add(full_path)
    
    # Get videos from cropped_videos subdirectory (if it exists)
    cropped_videos_dir = os.path.join(videos_dir, "cropped_videos")
    cropped_videos = find_videos_in_directory(cropped_videos_dir)
    for relative_path, full_path in cropped_videos:
        if full_path not in seen_paths:
            all_videos.append((relative_path, full_path))
            seen_paths.add(full_path)
    
    return all_videos


def interactive_video_selection():
    """Interactive video selection with numbered list."""
    all_videos = list_all_videos()
    
    if not all_videos:
        print("❌ No videos found in videos directory!")
        return None
    
    print("\n📹 Available videos:")
    print("-" * 80)
    
    for i, (relative_path, full_path) in enumerate(all_videos, 1):
        file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
        print(f"{i:2d}. {relative_path:<50} ({file_size:.1f} MB)")
    
    print("-" * 80)
    
    while True:
        try:
            choice = input(f"\nSelect video (1-{len(all_videos)}) or press Enter to cancel: ").strip()
            if not choice:
                return None
            
            video_index = int(choice) - 1
            if 0 <= video_index < len(all_videos):
                selected_video = all_videos[video_index][1]
                print(f"✅ Selected: {os.path.basename(selected_video)}")
                return selected_video
            else:
                print(f"❌ Please enter a number between 1 and {len(all_videos)}")
        except ValueError:
            print("❌ Please enter a valid number")


def display_video_list():
    """Display all available videos in a formatted list."""
    all_videos = list_all_videos()
    if all_videos:
        print("\n📹 Available videos:")
        print("-" * 80)
        for i, (relative_path, full_path) in enumerate(all_videos, 1):
            file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
            print(f"{i:2d}. {relative_path:<50} ({file_size:.1f} MB)")
        return True
    else:
        print("❌ No videos found!")
        return False 