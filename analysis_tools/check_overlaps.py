import pandas as pd

# Read ground truth
gt = pd.read_csv('ground_truth_template.csv', comment='#')

print("=== CHECKING FOR OVERLAPPING INTERVALS ===\n")

# Sort by start_frame for easier analysis
gt_sorted = gt.sort_values('start_frame').reset_index(drop=True)

print("All intervals (sorted by start_frame):")
for idx, row in gt_sorted.iterrows():
    print(f"{idx+1:2d}. Frames {row['start_frame']:5d}-{row['end_frame']:5d}")

print("\n=== OVERLAP ANALYSIS ===")

overlaps_found = False

for i in range(len(gt_sorted)):
    for j in range(i+1, len(gt_sorted)):
        interval1 = gt_sorted.iloc[i]
        interval2 = gt_sorted.iloc[j]
        
        # Check if intervals overlap
        # Overlap occurs if: start1 <= end2 AND start2 <= end1
        if (interval1['start_frame'] <= interval2['end_frame'] and 
            interval2['start_frame'] <= interval1['end_frame']):
            
            overlap_start = max(interval1['start_frame'], interval2['start_frame'])
            overlap_end = min(interval1['end_frame'], interval2['end_frame'])
            overlap_frames = overlap_end - overlap_start + 1
            
            print(f"❌ OVERLAP FOUND:")
            print(f"   Interval {i+1}: Frames {interval1['start_frame']}-{interval1['end_frame']}")
            print(f"   Interval {j+1}: Frames {interval2['start_frame']}-{interval2['end_frame']}")
            print(f"   Overlap: Frames {overlap_start}-{overlap_end} ({overlap_frames} frames)")
            print()
            overlaps_found = True

if not overlaps_found:
    print("✅ No overlapping intervals found!")

print("\n=== ADJACENT INTERVALS ===")
adjacent_count = 0
for i in range(len(gt_sorted)-1):
    current_end = gt_sorted.iloc[i]['end_frame']
    next_start = gt_sorted.iloc[i+1]['start_frame']
    
    if next_start == current_end + 1:
        print(f"Adjacent: Interval {i+1} ends at {current_end}, Interval {i+2} starts at {next_start}")
        adjacent_count += 1
    elif next_start <= current_end:
        print(f"Overlap: Interval {i+1} ends at {current_end}, Interval {i+2} starts at {next_start}")
    else:
        gap = next_start - current_end - 1
        print(f"Gap: {gap} frames between Interval {i+1} (ends {current_end}) and Interval {i+2} (starts {next_start})")

print(f"\nTotal adjacent intervals: {adjacent_count}")

print("\n=== INTERVAL STATISTICS ===")
total_frames = 0
for idx, row in gt_sorted.iterrows():
    frames_in_interval = row['end_frame'] - row['start_frame'] + 1
    total_frames += frames_in_interval
    print(f"Interval {idx+1}: {frames_in_interval} frames")

print(f"\nTotal frames covered: {total_frames}")
print(f"Video duration: {total_frames/54:.1f} seconds (assuming 54 FPS)") 