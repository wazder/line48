import os
import cv2
from ultralytics import YOLO

def draw_text_with_background(img, text, org, font, scale, text_color, bg_color, thickness=1, alpha=0.6):
    """Draw text with background rectangle."""
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    box_coords = ((x, y - text_height - 10), (x + text_width + 10, y + 10))
    overlay = img.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x + 5, y), font, scale, text_color, thickness, cv2.LINE_AA)

def load_model():
    """Load YOLO model from the models directory."""
    # Get the models directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, "models")
    
    # Try different model files in order of preference
    model_files = [
        "yolo11x.pt",
        "yolo12x.pt", 
        "yolov8x.pt",
        "suitcase.pt"
    ]
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            # Loading model - logging removed for cleaner output
            model_single = YOLO(model_path).to("cuda")
            # Model loaded successfully - logging removed for cleaner output
            return model_single
    
    # If no local model found, try to download yolo11x.pt
    # No local model found, attempting to download - logging removed for cleaner output
    model_single = YOLO("yolo11x.pt").to("cuda")
    # Model downloaded and loaded successfully - logging removed for cleaner output
    return model_single

def is_box_inside(box_a, box_b, threshold=0.8):
    """
    Returns True if box_a is at least `threshold` inside box_b.
    box_a, box_b: [x1, y1, x2, y2]
    threshold: float, e.g. 0.8 for 80%
    """
    # Calculate intersection
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    # Area of box_a
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    if area_a == 0:
        return False
    # Fraction of box_a inside box_b
    frac_inside = inter_area / area_a
    return frac_inside >= threshold
