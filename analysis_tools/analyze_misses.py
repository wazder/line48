import pandas as pd

# Read ground truth and log
gt = pd.read_csv('ground_truth_template.csv', comment='#')
log = pd.read_csv('logs/IMG_0015_blurred_log_20250727_192146.csv')

print("=== ANALYZING MISSED DETECTIONS ===\n")

# Find intervals where handbags and backpacks were missed
print("Intervals with missed handbags:")
print("-" * 40)
for idx, row in gt.iterrows():
    start, end = row['start_frame'], row['end_frame']
    expected_handbags = int(row['handbag'])
    expected_backpacks = int(row['backpack'])
    
    detected_handbags = log[(log['Class'] == 'handbag') & 
                           (log['Frame'] >= start) & 
                           (log['Frame'] <= end) & 
                           (log['Direction'] == 'IN')]
    detected_backpacks = log[(log['Class'] == 'backpack') & 
                            (log['Frame'] >= start) & 
                            (log['Frame'] <= end) & 
                            (log['Direction'] == 'IN')]
    
    found_handbags = len(set(detected_handbags['Object ID']))
    found_backpacks = len(set(detected_backpacks['Object ID']))
    
    if expected_handbags > found_handbags:
        print(f"Frames {start}-{end}: Expected {expected_handbags} handbags, found {found_handbags} (missed {expected_handbags - found_handbags})")
    
    if expected_backpacks > found_backpacks:
        print(f"Frames {start}-{end}: Expected {expected_backpacks} backpacks, found {found_backpacks} (missed {expected_backpacks - found_backpacks})")

print("\n=== VERY BRIEF DETECTIONS ANALYSIS ===")
# Check if any "very_brief" detections were in missed intervals
very_brief_handbags = log[(log['Class'] == 'handbag') & 
                          (log['Confidence'] == 'very_brief') & 
                          (log['Direction'] == 'IN')]

print(f"Total 'very_brief' handbags: {len(very_brief_handbags)}")
for _, row in very_brief_handbags.iterrows():
    frame = row['Frame']
    obj_id = row['Object ID']
    print(f"  Object {obj_id} at frame {frame}")

print("\n=== CONFIDENCE DISTRIBUTION ===")
for cls in ['handbag', 'backpack']:
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    confidence_counts = class_detections['Confidence'].value_counts()
    print(f"\n{cls.capitalize()} confidence distribution:")
    for confidence, count in confidence_counts.items():
        print(f"  {confidence}: {count}")

print("\n=== RECOMMENDATIONS ===")
print("1. Count 'very_brief' handbags as correct")
print("2. Lower frame thresholds for handbags")
print("3. Consider model fine-tuning for handbag detection")
print("4. Analyze video segments with missed detections") 