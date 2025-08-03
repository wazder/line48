from pycocotools.coco import COCO
import os

# --- CONFIG ---
TARGET_CLASSES = ["person", "backpack", "handbag", "suitcase"]
COCO_ROOT = "COCO-Training"
TRAIN_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
VAL_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

def check_class_counts(annotations_file, split_name):
    """Check how many images contain each class"""
    print(f"\n📊 Analyzing {split_name} split...")
    
    coco = COCO(annotations_file)
    
    # Get category IDs
    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    print(f"🎯 Target categories: {TARGET_CLASSES}")
    print(f"📊 Category IDs: {cat_ids}")
    
    # Check each class individually
    for i, class_name in enumerate(TARGET_CLASSES):
        cat_id = cat_ids[i]
        img_ids = coco.getImgIds(catIds=[cat_id])
        print(f"📈 {class_name} (ID {cat_id}): {len(img_ids)} images")
    
    # Check images with ANY of our target classes
    img_ids_any = coco.getImgIds(catIds=cat_ids)
    print(f"📈 Images with ANY target class: {len(img_ids_any)}")
    
    # Check images with ALL target classes (current logic)
    img_ids_all = set(coco.getImgIds(catIds=[cat_ids[0]]))
    for cat_id in cat_ids[1:]:
        img_ids_all = img_ids_all.intersection(set(coco.getImgIds(catIds=[cat_id])))
    print(f"📈 Images with ALL target classes: {len(img_ids_all)}")
    
    # Check total images in dataset
    all_img_ids = coco.getImgIds()
    print(f"📈 Total images in {split_name}: {len(all_img_ids)}")
    
    return len(img_ids_any)

def main():
    """Main function"""
    print("🔍 COCO Dataset Analysis")
    print("=" * 50)
    
    # Check train split
    train_count = check_class_counts(TRAIN_ANNOTATIONS, "train")
    
    # Check val split  
    val_count = check_class_counts(VAL_ANNOTATIONS, "val")
    
    print(f"\n📊 Summary:")
    print(f"🎯 Images with ANY target class: {train_count + val_count}")
    print(f"📈 This should be much higher than 346!")

if __name__ == "__main__":
    main() 