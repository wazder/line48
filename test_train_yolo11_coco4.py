from ultralytics import YOLO
import os
import yaml

# --- CONFIG ---
DATASET_PATH = "coco4_yolo_dataset_coco_api/data.yaml"  # Path to your prepared dataset (will be created by the preparation script)
MODEL_SIZE = "yolo11n"  # Smallest model for testing
EPOCHS = 1  # Minimum epochs for testing
BATCH_SIZE = 6  # Minimum batch size for testing
IMG_SIZE = 320  # Smaller image size for testing
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937

def main():
    """Train YOLO11 model on COCO 4-class dataset - MINIMAL TEST VERSION"""
    print("🧪 Starting MINIMAL YOLO11 training test")
    print(f"📊 Model: {MODEL_SIZE}")
    print(f"🎯 Classes: person, backpack, handbag, suitcase")
    print(f"⚙️  Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🖼️  Image size: {IMG_SIZE}")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("Please run prepare_coco_subset_coco_api.py first")
        return
    
    # Load dataset info
    with open(DATASET_PATH, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    print(f"📁 Dataset path: {dataset_info['path']}")
    print(f"🎯 Number of classes: {dataset_info['nc']}")
    print(f"📝 Class names: {dataset_info['names']}")
    
    # Initialize model
    print(f"\n🔄 Loading {MODEL_SIZE} model...")
    model = YOLO(f"{MODEL_SIZE}.pt")
    
    # Training configuration - MINIMAL for testing
    training_args = {
        'data': DATASET_PATH,
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE,
        'lr0': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': MOMENTUM,
        'patience': 1,  # Minimum patience
        'save': True,
        'save_period': 1,  # Save every epoch
        'cache': False,  # Disable caching for testing
        'device': 0,  # Use GPU 0
        'workers': 1,  # Minimum workers
        'project': 'yolo11_coco4_test',
        'name': f'{MODEL_SIZE}_coco4_test_epochs{EPOCHS}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',  # Simple optimizer for testing
        'cos_lr': False,  # Disable cosine LR for testing
        'close_mosaic': 0,  # Disable mosaic
        'amp': False,  # Disable mixed precision for testing
        'overlap_mask': False,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': False,  # Disable plots for testing
        'save_txt': False,  # Disable text saves for testing
        'save_conf': False,  # Disable confidence saves for testing
        'save_crop': False,  # Disable crop saves for testing
        'conf': 0.001,  # Confidence threshold for validation
        'iou': 0.6,  # NMS IoU threshold
        'max_det': 300,  # Maximum detections per image
        'half': False,  # Disable FP16 for testing
        'dnn': False,  # Use OpenCV DNN for ONNX inference
        'plots': False,  # Disable training plots for testing
    }
    
    print(f"\n🧪 Starting MINIMAL training test with {EPOCHS} epochs...")
    print(f"📊 Training arguments: {training_args}")
    
    # Start training
    try:
        results = model.train(**training_args)
        
        print(f"\n✅ MINIMAL training test completed successfully!")
        print(f"📁 Results saved in: {results.save_dir}")
        print(f"📈 Training completed without errors!")
        
        # Try to get metrics safely
        try:
            if hasattr(results, 'results_dict'):
                map_value = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
                print(f"📊 Final mAP: {map_value}")
            else:
                print(f"📊 Training metrics available in results object")
        except Exception as e:
            print(f"📊 Training completed (metrics access: {e})")
        
        # Validate the trained model
        print(f"\n🔍 Validating trained model...")
        try:
            val_results = model.val()
            print(f"✅ Validation completed!")
            
            # Try to get validation metrics safely
            if hasattr(val_results, 'results_dict'):
                print(f"📊 Validation metrics: {val_results.results_dict}")
            else:
                print(f"📊 Validation completed successfully")
        except Exception as e:
            print(f"⚠️  Validation completed with minor issues: {e}")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return
    
    print(f"\n🧪 MINIMAL training test completed!")
    print(f"📁 Check the 'yolo11_coco4_test' folder for results")

if __name__ == "__main__":
    main() 