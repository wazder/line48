"""
SAM-Only Utilities for LineLogic
Pure SAM-based segmentation without YOLO dependency.
"""

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from typing import List, Tuple, Optional, Dict
import os

class SAMOnlyLineLogic:
    """
    Pure SAM-based system for precise segmentation and line crossing detection.
    No YOLO dependency - uses SAM's automatic mask generation.
    """
    
    def __init__(self, 
                 sam_model_type: str = "vit_h",
                 sam_checkpoint: str = None,
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SAM-only system.
        
        Args:
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            sam_checkpoint: Path to SAM checkpoint
            device: Device to run models on
        """
        self.device = device
        self.sam_model_type = sam_model_type
        
        # Initialize SAM
        self._load_sam_model(sam_checkpoint)
        
        # Object size and shape filters (to identify person, backpack, handbag, suitcase)
        self.object_filters = {
            'person': {
                'min_area': 3000,    # Minimum pixel area
                'max_area': 50000,   # Maximum pixel area
                'min_aspect_ratio': 0.3,  # height/width
                'max_aspect_ratio': 3.0,
                'class_name': 'person'
            },
            'backpack': {
                'min_area': 800,
                'max_area': 8000,
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 2.0,
                'class_name': 'backpack'
            },
            'handbag': {
                'min_area': 500,
                'max_area': 5000,
                'min_aspect_ratio': 0.4,
                'max_aspect_ratio': 2.5,
                'class_name': 'handbag'
            },
            'suitcase': {
                'min_area': 1500,
                'max_area': 15000,
                'min_aspect_ratio': 0.4,
                'max_aspect_ratio': 2.5,
                'class_name': 'suitcase'
            }
        }
        
        # Tracking system
        self.track_id_counter = 0
        self.active_tracks = {}  # Track previous frame masks for consistency
        
        # Print GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU ENABLED: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"🎯 SAM-Only LineLogic initialized on {device}")
        else:
            print(f"⚠️ GPU NOT AVAILABLE - Running on CPU")
            print(f"🎯 SAM-Only LineLogic initialized on {device}")
    
    def _load_sam_model(self, checkpoint_path: Optional[str] = None):
        """Load SAM model and automatic mask generator."""
        
        # Default checkpoint paths
        if checkpoint_path is None:
            checkpoint_files = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth", 
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            checkpoint_path = checkpoint_files[self.sam_model_type]
            
            # Check if model exists
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
            
            # Create automatic mask generator
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,          # Grid density
                pred_iou_thresh=0.7,         # Quality threshold
                stability_score_thresh=0.8,  # Stability threshold
                crop_n_layers=1,             # Crop layers
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,    # Minimum region size
            )
            
            print(f"✅ SAM-Only model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load SAM model: {e}")
            raise
    
    def detect_and_segment(self, frame: np.ndarray, frame_count: int = 0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect and segment objects using pure SAM approach.
        
        Args:
            frame: Input frame (BGR format)
            frame_count: Current frame number
            
        Returns:
            Tuple of (segmented_frame, detection_results)
        """
        # Convert BGR to RGB for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate masks automatically
        masks = self.mask_generator.generate(rgb_frame)
        
        if not masks:
            return frame, []
        
        # Filter and classify masks
        detection_results = []
        segmented_frame = frame.copy()
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']  # x, y, w, h format
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            
            # Convert bbox to x1, y1, x2, y2 format
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Classify object based on size and shape
            object_class = self._classify_mask(mask, area, bbox)
            
            if object_class:
                # Create tracking ID (simple approach for now)
                track_id = self._get_track_id(mask, frame_count)
                
                detection_result = {
                    'bbox': [x1, y1, x2, y2],
                    'class': object_class,
                    'confidence': stability_score,  # Use stability score as confidence
                    'class_id': self._get_class_id(object_class),
                    'track_id': track_id,
                    'mask': mask,
                    'mask_score': stability_score
                }
                
                detection_results.append(detection_result)
                
                # Visualize mask on frame
                colored_mask = self._create_colored_mask(mask, i)
                segmented_frame = cv2.addWeighted(segmented_frame, 0.7, colored_mask, 0.3, 0)
                
                # Draw bounding box
                cv2.rectangle(segmented_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f"{object_class} T:{track_id} ({stability_score:.2f})"
                cv2.putText(segmented_frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Frame {frame_count}: Found {len(detection_results)} objects")
        
        return segmented_frame, detection_results
    
    def _classify_mask(self, mask: np.ndarray, area: int, bbox: List[int]) -> Optional[str]:
        """
        Classify mask based on size and shape characteristics.
        
        Args:
            mask: Binary mask
            area: Mask area in pixels
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Object class name or None
        """
        x, y, w, h = bbox
        aspect_ratio = h / w if w > 0 else 0
        
        # Check each object type
        for obj_type, filters in self.object_filters.items():
            if (filters['min_area'] <= area <= filters['max_area'] and
                filters['min_aspect_ratio'] <= aspect_ratio <= filters['max_aspect_ratio']):
                return obj_type
        
        return None  # Unclassified object
    
    def _get_class_id(self, class_name: str) -> int:
        """Get COCO-like class ID for compatibility."""
        class_ids = {
            'person': 0,
            'backpack': 24,
            'handbag': 26,
            'suitcase': 28
        }
        return class_ids.get(class_name, -1)
    
    def _get_track_id(self, mask: np.ndarray, frame_count: int) -> int:
        """
        Simple tracking by mask overlap with previous frame.
        
        Args:
            mask: Current mask
            frame_count: Current frame number
            
        Returns:
            Track ID
        """
        # Simple approach: assign new ID for each mask
        # TODO: Implement proper tracking based on mask overlap
        self.track_id_counter += 1
        return self.track_id_counter
    
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