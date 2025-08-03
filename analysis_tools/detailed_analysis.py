import pandas as pd

# Read ground truth and log
gt = pd.read_csv('ground_truth_template.csv', comment='#')
log = pd.read_csv('logs/IMG_0015_blurred_log_20250727_192146.csv')

print("=== DETAILED OBJECT TRACKING ANALYSIS ===\n")

# Get all unique object IDs and their frame ranges
object_tracking = {}
for _, row in log.iterrows():
    obj_id = row['Object ID']
    frame = row['Frame']
    if obj_id not in object_tracking:
        object_tracking[obj_id] = {'frames': [], 'class': row['Class']}
    object_tracking[obj_id]['frames'].append(frame)

# Sort frames for each object
for obj_id in object_tracking:
    object_tracking[obj_id]['frames'].sort()

print("Objects that appear in multiple intervals:")
print("=" * 60)

# Check which objects appear in multiple intervals
for obj_id, data in object_tracking.items():
    frames = data['frames']
    class_name = data['class']
    
    # Find which intervals this object appears in
    intervals_found = []
    for idx, gt_row in gt.iterrows():
        start, end = gt_row['start_frame'], gt_row['end_frame']
        # Check if any frame of this object falls in this interval
        if any(start <= frame <= end for frame in frames):
            intervals_found.append((idx+1, start, end))
    
    if len(intervals_found) > 1:
        print(f"Object {obj_id} ({class_name}):")
        print(f"  Frames: {frames}")
        print(f"  Appears in {len(intervals_found)} intervals:")
        for interval_num, start, end in intervals_found:
            print(f"    Interval {interval_num}: Frames {start}-{end}")
        print()

print("\n=== SPECIFIC EXAMPLES ===")

# Look at some specific problematic cases
problematic_objects = []
for obj_id, data in object_tracking.items():
    frames = data['frames']
    if len(frames) > 1:
        frame_span = max(frames) - min(frames)
        if frame_span > 1000:  # Objects tracked for more than 1000 frames
            problematic_objects.append((obj_id, data, frame_span))

print("Objects tracked for very long periods:")
for obj_id, data, span in sorted(problematic_objects, key=lambda x: x[2], reverse=True)[:10]:
    print(f"Object {obj_id} ({data['class']}): {span} frames ({span/54:.1f}s)")
    print(f"  Frame range: {min(data['frames'])} - {max(data['frames'])}")
    print()

print("=== INTERVAL-BY-INTERVAL OBJECT DISTRIBUTION ===")
for idx, gt_row in gt.iterrows():
    start, end = gt_row['start_frame'], gt_row['end_frame']
    print(f"\nInterval {idx+1}: Frames {start}-{end}")
    
    # Find all objects in this interval
    objects_in_interval = []
    for obj_id, data in object_tracking.items():
        if any(start <= frame <= end for frame in data['frames']):
            objects_in_interval.append((obj_id, data['class']))
    
    # Group by class
    by_class = {}
    for obj_id, class_name in objects_in_interval:
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(obj_id)
    
    for class_name, obj_ids in by_class.items():
        print(f"  {class_name}: {len(obj_ids)} objects (IDs: {sorted(obj_ids)})")

print("\n=== SUMMARY ===")
print("The issue is likely that objects are being tracked across multiple intervals")
print("because they remain in the scene longer than expected, causing them to be")
print("counted in multiple intervals when they should only be counted once.") 