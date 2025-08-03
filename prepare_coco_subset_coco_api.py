import os
import shutil
import yaml
from pycocotools.coco import COCO
from tqdm import tqdm
import json

# --- CONFIG ---
TARGET_CLASSES = ["person", "backpack", "handbag", "suitcase"]
EXPORT_DIR = "coco4_yolo_dataset_coco_api"

# Paths to your downloaded COCO files
COCO_ROOT = "COCO-Training"  # Your COCO dataset root directory
TRAIN_IMAGES_DIR = os.path.join(COCO_ROOT, "train2017")
VAL_IMAGES_DIR = os.path.join(COCO_ROOT, "val2017")
TRAIN_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
VAL_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox format (x, y, width, height) to YOLO format (center_x, center_y, width, height)"""
    x, y, w, h = bbox
    
    # Convert to center coordinates and normalize
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return center_x, center_y, width, height

def process_split(split_name, images_dir, annotations_file, export_dir):
    """Process a single split (train or val)"""
    print(f"\n🔄 Processing {split_name} split...")
    
    # Load COCO annotations
    print(f"📖 Loading annotations from {annotations_file}")
    coco = COCO(annotations_file)
    
    # Get category IDs for target classes
    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    print(f"🎯 Target categories: {TARGET_CLASSES}")
    print(f"📊 Category IDs: {cat_ids}")
    
    # Get image IDs that contain ANY of our target classes
    # We need to get images for each class separately and combine them
    all_img_ids = set()
    for cat_id in cat_ids:
        class_img_ids = coco.getImgIds(catIds=[cat_id])
        all_img_ids.update(class_img_ids)
    
    img_ids = list(all_img_ids)
    print(f"🖼️  Found {len(img_ids)} images containing ANY target class")
    
    # Create output directories
    img_out_dir = os.path.join(export_dir, split_name, "images")
    lbl_out_dir = os.path.join(export_dir, split_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    
    # Process each image
    successful_exports = 0
    skipped_images = 0
    
    for img_id in tqdm(img_ids, desc=f"Processing {split_name}"):
        try:
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            img_path = os.path.join(images_dir, img_filename)
            
            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                skipped_images += 1
                continue
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            
            # Convert annotations to YOLO format
            yolo_labels = []
            img_width = img_info['width']
            img_height = img_info['height']
            
            for ann in anns:
                # Get class ID (0-3 for our 4 classes)
                class_name = coco.loadCats(ann['category_id'])[0]['name']
                if class_name in TARGET_CLASSES:
                    class_id = TARGET_CLASSES.index(class_name)
                    
                    # Convert bbox to YOLO format
                    bbox = ann['bbox']  # [x, y, width, height]
                    center_x, center_y, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
                    
                    # Add to YOLO labels
                    yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Only export if we have target class annotations
            if yolo_labels:
                # Copy image
                img_dst = os.path.join(img_out_dir, img_filename)
                shutil.copy2(img_path, img_dst)
                
                # Write label file
                label_filename = img_filename.replace('.jpg', '.txt')
                label_path = os.path.join(lbl_out_dir, label_filename)
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                successful_exports += 1
            else:
                skipped_images += 1
                
        except Exception as e:
            print(f"❌ Error processing image {img_id}: {e}")
            skipped_images += 1
            continue
    
    print(f"✅ {split_name}: Successfully exported {successful_exports} images, skipped {skipped_images}")
    return successful_exports

def create_data_yaml(export_dir, train_count, val_count):
    """Create data.yaml file for YOLO training"""
    data_yaml = {
        'path': os.path.abspath(export_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES
    }
    
    yaml_path = os.path.join(export_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"📄 Created data.yaml with {len(TARGET_CLASSES)} classes: {TARGET_CLASSES}")
    print(f"📊 Dataset stats: {train_count} train images, {val_count} val images")

def main():
    """Main function"""
    print("🚀 Starting COCO dataset preparation for YOLO training")
    print(f"🎯 Target classes: {TARGET_CLASSES}")
    print(f"📁 Export directory: {EXPORT_DIR}")
    
    # Check if required files exist
    required_files = [
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TRAIN_ANNOTATIONS,
        VAL_ANNOTATIONS
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file/directory not found: {file_path}")
            print("Please make sure you have downloaded and extracted:")
            print("- train2017.zip")
            print("- val2017.zip") 
            print("- annotations_trainval2017.zip")
            return
    
    # Clean export directory
    if os.path.exists(EXPORT_DIR):
        print(f"🧹 Cleaning existing export directory: {EXPORT_DIR}")
        shutil.rmtree(EXPORT_DIR)
    
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Process train split
    train_count = process_split("train", TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS, EXPORT_DIR)
    
    # Process val split
    val_count = process_split("val", VAL_IMAGES_DIR, VAL_ANNOTATIONS, EXPORT_DIR)
    
    # Create data.yaml
    create_data_yaml(EXPORT_DIR, train_count, val_count)
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"📁 Output directory: {EXPORT_DIR}")
    print(f"📊 Total images: {train_count + val_count}")
    print(f"🎯 Classes: {TARGET_CLASSES}")
    print(f"📄 Configuration: {os.path.join(EXPORT_DIR, 'data.yaml')}")

if __name__ == "__main__":
    main() 