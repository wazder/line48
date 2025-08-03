"""
Configuration file for LineLogic
Contains all the default configuration values and utility functions.
"""

import os
import supervision as sv

# Default video paths (will be overridden during runtime)
SOURCE_VIDEO_PATH = ""
TARGET_VIDEO_PATH = ""
LOG_CSV_PATH = ""

# Line configuration defaults
LINE_HEIGHT = 1080
LINE_POINTS = [
    sv.Point(880, 0),   # Line 1
    sv.Point(960, 0),   # Line 2
    sv.Point(1040, 0),  # Line 3
    sv.Point(1120, 0)   # Line 4
]

# Line IDs for identification
LINE_IDS = [1, 2, 3, 4]

# COCO class names for detection
COCO_NAMES = ["person", "backpack", "handbag", "suitcase"]

def get_next_filename(base_name, extension):
    """
    Generate a unique filename by appending a number if the file already exists.
    
    Args:
        base_name: Base name without extension
        extension: File extension (e.g., '.mp4', '.csv')
    
    Returns:
        str: Unique filename
    """
    counter = 1
    filename = f"{base_name}{extension}"
    
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1
    
    return filename 