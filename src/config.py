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
BASE_X = 960
LINE_SPACING = 125
LINE_HEIGHT = 1080

# Line margins (top and bottom spacing)
LINE_TOP_MARGIN = 150      # 150px from top
LINE_BOTTOM_MARGIN = 150   # 150px from bottom
LINE_START_Y = LINE_TOP_MARGIN
LINE_END_Y = LINE_HEIGHT - LINE_BOTTOM_MARGIN

LINE_POINTS = [
    sv.Point(BASE_X - 2*(LINE_SPACING), LINE_START_Y), #LeftMost line
    sv.Point(BASE_X - LINE_SPACING, LINE_START_Y),     # Left line
    sv.Point(BASE_X, LINE_START_Y),                    # Center line
    sv.Point(BASE_X + LINE_SPACING, LINE_START_Y),     # Right line
    sv.Point(BASE_X + (2*LINE_SPACING), LINE_START_Y), # RightMost line
]

# Line IDs for identification
LINE_IDS = ["LeftMost", "Left", "Center", "Right", "RightMost"]

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