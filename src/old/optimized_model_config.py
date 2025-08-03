import sys
import os
from ultralytics import YOLO

# Add virtual environment to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, "envs", "lov10-env310", "Lib", "site-packages")
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

def load_optimized_model():
    """Load YOLO model with optimized parameters for handbag/backpack detection."""
    
    # Load model
    model_path = os.path.join(project_root, "models", "yolo11x.pt")
    if not os.path.exists(model_path):
        print("⚠️ yolo11x.pt not found, downloading...")
        model_single = YOLO("yolo11x.pt")
    else:
        model_single = YOLO(model_path)
    
    # Move to GPU
    model_single = model_single.to("cuda")
    
    # OPTIMIZED PARAMETERS FOR BETTER DETECTION
    
    # 1. Lower confidence threshold (more detections)
    model_single.conf = 0.25  # Default is 0.5, lower = more detections
    
    # 2. Lower NMS threshold (keep more overlapping detections)
    model_single.iou = 0.45   # Default is 0.7, lower = more detections
    
    # 3. Enable augmented inference
    model_single.augment = True
    
    # 4. Use larger image size for better detection of small objects
    model_single.imgsz = 1280  # Default is 640, larger = better for small objects
    
    print("🔧 Optimized model parameters:")
    print(f"   Confidence threshold: {model_single.conf}")
    print(f"   NMS threshold: {model_single.iou}")
    print(f"   Augmented inference: {model_single.augment}")
    print(f"   Image size: {model_single.imgsz}")
    print(f"   Device: {model_single.device}")
    
    return model_single

def get_class_specific_parameters():
    """Get class-specific parameters for post-processing."""
    
    # Class-specific confidence thresholds
    class_conf_thresholds = {
        'person': 0.4,      # Keep reasonable for persons
        'backpack': 0.2,    # Lower for backpacks
        'handbag': 0.15,    # Even lower for handbags
        'suitcase': 0.25    # Medium for suitcases
    }
    
    # Class-specific NMS thresholds
    class_nms_thresholds = {
        'person': 0.6,      # Standard
        'backpack': 0.4,    # More lenient
        'handbag': 0.35,    # Very lenient
        'suitcase': 0.5     # Medium
    }
    
    return class_conf_thresholds, class_nms_thresholds

def apply_class_specific_filtering(detections, class_names):
    """Apply class-specific filtering to detections."""
    
    conf_thresholds, nms_thresholds = get_class_specific_parameters()
    
    # Filter detections based on class-specific confidence
    filtered_detections = []
    
    for i, conf in enumerate(detections.confidence):
        class_id = detections.class_id[i]
        class_name = class_names[class_id]
        
        # Get class-specific threshold
        threshold = conf_thresholds.get(class_name, 0.25)
        
        if conf >= threshold:
            filtered_detections.append(i)
    
    # Return filtered detections
    if filtered_detections:
        return detections[filtered_detections]
    else:
        return detections[[]]  # Empty detections

# Example usage
if __name__ == "__main__":
    model = load_optimized_model()
    conf_thresholds, nms_thresholds = get_class_specific_parameters()
    
    print("\n📊 Class-specific parameters:")
    for cls in ['person', 'backpack', 'handbag', 'suitcase']:
        print(f"   {cls}: conf={conf_thresholds[cls]}, nms={nms_thresholds[cls]}") 