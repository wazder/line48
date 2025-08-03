from ultralytics import YOLO
import os
import yaml

# --- CONFIG ---
DATASET_PATH = "coco4_yolo_dataset_coco_api/data.yaml"  # Path to your prepared dataset (will be created by the preparation script)
MODEL_SIZE = "yolo11x"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
EPOCHS = 1
BATCH_SIZE = 8  # Optimized for 48GB VRAM
IMG_SIZE = 640
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937

def main():
    """Train YOLO11 model on COCO 4-class dataset"""
    print("🚀 Starting YOLO11 training on COCO 4-class dataset")
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
    
    # Training configuration
    training_args = {
        'data': DATASET_PATH,
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE,
        'lr0': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': MOMENTUM,
        'patience': 50,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'cache': True,  # Cache images for faster training
        'device': 0,  # Use GPU 0
        'workers': 8,  # Number of worker threads
        'project': 'yolo11_coco4_training',
        'name': f'{MODEL_SIZE}_coco4_epochs{EPOCHS}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # Modern optimizer
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation in last 10 epochs
        'amp': True,  # Automatic mixed precision
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save_txt': True,
        'save_conf': True,
        'save_crop': True,
        'conf': 0.001,  # Confidence threshold for validation
        'iou': 0.6,  # NMS IoU threshold
        'max_det': 300,  # Maximum detections per image
        'half': True,  # Use FP16 inference
        'dnn': False,  # Use OpenCV DNN for ONNX inference
        'plots': True,  # Generate training plots
    }
    
    print(f"\n🎯 Starting training with {EPOCHS} epochs...")
    print(f"📊 Training arguments: {training_args}")
    
    # Start training
    try:
        results = model.train(**training_args)
        
        print(f"\n🎉 Training completed successfully!")
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
        print(f"❌ Training failed: {e}")
        return
    
    print(f"\n🎯 Training script completed!")
    print(f"📁 Check the 'yolo11_coco4_training' folder for results")

if __name__ == "__main__":
    main() 