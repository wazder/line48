import pandas as pd

# User: set these paths as needed
GROUND_TRUTH_CSV = 'ground_truth_template.csv'
FRAME_LOG_CSV = 'logs\IMG_0015_blurred_log_20250727_192146.csv'

# Read ground truth
gt = pd.read_csv(GROUND_TRUTH_CSV, comment='#')
# Read frame-based logic log
log = pd.read_csv(FRAME_LOG_CSV)

# Classes to check
CLASSES = ['backpack', 'person', 'handbag', 'suitcase']

print(f"Comparing {FRAME_LOG_CSV} to {GROUND_TRUTH_CSV}\n")

totals = {cls: {'overcounted': 0, 'missed': 0} for cls in CLASSES}

for idx, row in gt.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    print(f"Frames {start}-{end}:")
    for cls in CLASSES:
        expected = int(row[cls])
        detected = log[(log['Class'] == cls) & (log['Frame'] >= start) & (log['Frame'] <= end) & (log['Direction'] == 'IN')]
        unique_ids = set(detected['Object ID'])
        found = len(unique_ids)
        if found == expected:
            status = '✅ Correct'
        elif found > expected:
            status = f'❌ Overcounted (+{found-expected})'
            totals[cls]['overcounted'] += (found-expected)
        else:
            status = f'❌ Missed ({expected-found})'
            totals[cls]['missed'] += (expected-found)
        print(f"  {cls}: expected {expected}, found {found} {status}")
    print()

print("Summary across all intervals:")
print(f"{'Class':<10} {'Overcounted':<12} {'Missed':<8}")
for cls in CLASSES:
    print(f"{cls:<10} {totals[cls]['overcounted']:<12} {totals[cls]['missed']:<8}")

# Calculate total expected and detected for validation
print("\n=== VALIDATION ===")
gt_totals = {}
log_totals = {}
for cls in CLASSES:
    gt_totals[cls] = gt[cls].sum()
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    log_totals[cls] = len(set(class_detections['Object ID']))

print("Class      | Expected | Detected | Overcounted | Missed")
print("-" * 55)
for cls in CLASSES:
    expected = gt_totals[cls]
    detected = log_totals[cls]
    overcounted = totals[cls]['overcounted']
    missed = totals[cls]['missed']
    print(f"{cls:<10} | {expected:<8} | {detected:<8} | {overcounted:<11} | {missed:<6}")

print("Done.") 