"""
SAM (Segment Anything Model) Utilities for LineLogic
Integrates SAM with LineLogic for pixel-perfect segmentation and line crossing detection.
"""

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import supervision as sv
from typing import List, Tuple, Optional, Dict
import os

class SAMLineLogic:
    """
    SAM + LineLogic hybrid system for precise segmentation and line crossing detection.
    
    Architecture:
    1. YOLO detects objects and provides bounding boxes
    2. SAM uses YOLO boxes as prompts for precise segmentation
    3. LineLogic tracks segments and detects line crossings
    4. Frame-based validation ensures reliability
    """
    
    def __init__(self, 
                 sam_model_type: str = "vit_h",
                 sam_checkpoint: str = None,
                 yolo_model: str = "yolo11s.pt",
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SAM + LineLogic system.
        
        Args:
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            sam_checkpoint: Path to SAM checkpoint
            yolo_model: YOLO model name or path
            device: Device to run models on
        """
        self.device = device
        self.sam_model_type = sam_model_type
        
        # Initialize YOLO for object detection with tracking
        self.yolo_model = YOLO(yolo_model)
        
        # Initialize SAM
        self._load_sam_model(sam_checkpoint)
        
        # Target classes for detection
        self.target_classes = ["person", "backpack", "handbag", "suitcase"]
        self.class_ids = [0, 24, 26, 28]  # COCO class IDs
        
        # Class-specific confidence thresholds - lowered person and handbag thresholds
        self.confidence_thresholds = {
            0: 0.80,   # person - lowered from 0.92 to detect more
            24: 0.70,  # backpack - target: 3 
            26: 0.65,  # handbag - lowered from 0.75 to detect more
            28: 0.70   # suitcase - target: 2 
        }
        
        print(f"🎯 SAM + LineLogic initialized on {device}")
    
    def _load_sam_model(self, checkpoint_path: Optional[str] = None):
        """Load SAM model and predictor."""
        
        # Default checkpoint paths
        if checkpoint_path is None:
            checkpoint_files = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth", 
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            checkpoint_path = checkpoint_files[self.sam_model_type]
            
            # Check if model exists, if not provide download info
            if not os.path.exists(checkpoint_path):
                download_urls = {
                    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                }
                print(f"⚠️ SAM checkpoint not found: {checkpoint_path}")
                print(f"📥 Download from: {download_urls[self.sam_model_type]}")
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        
        # Load SAM model
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
        except Exception as e:
            print(f"❌ Failed to load SAM model: {e}")
            raise
    
    def detect_and_segment(self, frame: np.ndarray, frame_count: int = 0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects with YOLO and segment with SAM.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (segmented_frame, detection_results)
        """
        # YOLO detection with tracking
        yolo_results = self.yolo_model.predict(frame, verbose=False)
        
        if len(yolo_results) == 0:
            return frame, []
        
        # Extract detections
        detections = yolo_results[0]
        boxes = detections.boxes
        
        if boxes is None or len(boxes) == 0:
            return frame, []
        
        # Filter for target classes
        detection_results = []
        valid_boxes = []
        
        # Count valid detections but don't print here (will be printed in line crossing logic)
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Use class-specific confidence threshold
            required_confidence = self.confidence_thresholds.get(class_id, 0.5)
            if class_id in self.class_ids and confidence > required_confidence:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_name = self.yolo_model.names[class_id]
                
                # Add track_id for line crossing detection
                track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None and len(box.id) > 0 else i
                
                # Detection logged (will show in line crossing if crosses line)
                
                detection_results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': confidence,
                    'class_id': class_id,
                    'track_id': track_id  # Add track_id for line crossing
                })
                valid_boxes.append([x1, y1, x2, y2])
        
        # Don't print detection count here (will be shown in line crossing summary)
        
        if not valid_boxes:
            return frame, []
        
        # SAM segmentation
        segmented_frame = frame.copy()
        
        # Set image for SAM predictor
        self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Generate masks for each detection
        for i, (detection, bbox) in enumerate(zip(detection_results, valid_boxes)):
            # Use bounding box as prompt
            input_box = np.array(bbox)
            
            try:
                # Predict mask
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                if len(masks) > 0:
                    mask = masks[0]  # Use first (and only) mask
                    mask_area = np.sum(mask)
                    
                    # Filter out very small masks (noise) - VERY RELAXED
                    min_mask_area = 5  # Extremely low threshold for more detections
                    if mask_area < min_mask_area:
                        print(f"      ⚠️ Mask too small: {mask_area} < {min_mask_area}")
                        continue

                    detection['mask'] = mask
                    detection['mask_score'] = float(scores[0])
                    
                    # SAM mask created - logging removed for cleaner output
                    
                    # Visualize mask on frame
                    colored_mask = self._create_colored_mask(mask, i)
                    segmented_frame = cv2.addWeighted(segmented_frame, 0.7, colored_mask, 0.3, 0)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(segmented_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{detection['class']}: {detection['confidence']:.2f}"
                    cv2.putText(segmented_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            except Exception as e:
                detection['mask'] = None
                detection['mask_score'] = 0.0
        
        return segmented_frame, detection_results
    
    def _create_colored_mask(self, mask: np.ndarray, color_index: int) -> np.ndarray:
        """Create colored mask overlay."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        color = colors[color_index % len(colors)]
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask] = color
        
        return colored_mask
    
    def get_mask_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get centroid of segmentation mask."""
        if mask is None:
            return None, None
            
        # Find mask coordinates
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0:
            return None, None
            
        # Calculate centroid
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        return centroid_x, centroid_y
    
    def get_mask_area(self, mask: np.ndarray) -> int:
        """Get area of segmentation mask in pixels."""
        if mask is None:
            return 0
        return int(np.sum(mask))
    
    def mask_intersects_line(self, mask: np.ndarray, line_points: List[sv.Point]) -> bool:
        """Check if segmentation mask intersects with a line."""
        if mask is None or len(line_points) != 2:
            return False
            
        # Create line mask
        line_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.line(line_mask, 
                (int(line_points[0].x), int(line_points[0].y)),
                (int(line_points[1].x), int(line_points[1].y)),
                1, thickness=5)  # Thick line for better intersection detection
        
        # Check intersection
        intersection = np.logical_and(mask, line_mask)
        return np.any(intersection)

def load_sam_model(checkpoint_path: str):
    """
    Load SAM model from checkpoint.
    
    Args:
        checkpoint_path: Path to SAM checkpoint file
        
    Returns:
        SAM predictor object
    """
    try:
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(sam)
        print(f"✅ SAM model loaded from: {checkpoint_path}")
        return predictor
    except Exception as e:
        print(f"❌ Failed to load SAM model: {e}")
        raise

def download_sam_model(model_type: str = "vit_b") -> str:
    """
    Download SAM model checkpoint.
    
    Args:
        model_type: Model type ('vit_h', 'vit_l', 'vit_b')
        
    Returns:
        Path to downloaded checkpoint
    """
    import urllib.request
    
    download_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_files = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    
    url = download_urls[model_type]
    filename = checkpoint_files[model_type]
    
    if os.path.exists(filename):
        print(f"✅ SAM model already exists: {filename}")
        return filename
    
    print(f"📥 Downloading SAM model: {model_type}")
    print(f"   URL: {url}")
    print(f"   File: {filename}")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Download completed: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise