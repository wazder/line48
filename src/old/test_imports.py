"""
Test script to verify that all imports work correctly after environment reorganization.
"""

import sys
import os

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)
    print(f"✅ Added environment path: {env_path}")
else:
    print(f"❌ Environment path not found: {env_path}")

print("\n🧪 Testing imports...")

try:
    import supervision as sv
    print("✅ supervision imported successfully")
except ImportError as e:
    print(f"❌ supervision import failed: {e}")

try:
    import ultralytics
    print("✅ ultralytics imported successfully")
except ImportError as e:
    print(f"❌ ultralytics import failed: {e}")

try:
    import cv2
    print("✅ opencv-python imported successfully")
except ImportError as e:
    print(f"❌ opencv-python import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    from config import SOURCE_VIDEO_PATH
    print("✅ local config imported successfully")
except ImportError as e:
    print(f"❌ local config import failed: {e}")

try:
    from utils import load_model
    print("✅ local utils imported successfully")
except ImportError as e:
    print(f"❌ local utils import failed: {e}")

print("\n🎯 Import test complete!") 