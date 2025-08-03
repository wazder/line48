import pandas as pd

# User: set these paths as needed
GROUND_TRUTH_CSV = 'ground_truth_template.csv'
FRAME_LOG_CSV = 'logs/IMG_0015_blurred_log_20250727_192146.csv'  # Update if needed

# Read ground truth
gt = pd.read_csv(GROUND_TRUTH_CSV, comment='#')
# Read frame-based logic log
log = pd.read_csv(FRAME_LOG_CSV)

# Classes to check
CLASSES = ['backpack', 'person', 'handbag', 'suitcase']

print(f"Comparing {FRAME_LOG_CSV} to {GROUND_TRUTH_CSV}\n")

# Debug: Print total expected counts from ground truth
print("=== GROUND TRUTH TOTALS ===")
gt_totals = {}
for cls in CLASSES:
    total = gt[cls].sum()
    gt_totals[cls] = total
    print(f"{cls}: {total}")
print()

# Debug: Print total detected counts from log
print("=== LOG TOTALS ===")
log_totals = {}
for cls in CLASSES:
    # Count all IN detections for this class
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    unique_ids = set(class_detections['Object ID'])
    log_totals[cls] = len(unique_ids)
    print(f"{cls}: {len(unique_ids)} unique IDs")
print()

totals = {cls: {'correct': 0, 'overcounted': 0, 'missed': 0} for cls in CLASSES}

print("=== INTERVAL-BY-INTERVAL ANALYSIS ===")
for idx, row in gt.iterrows():
    start, end = int(row['start_frame']), int(row['end_frame'])
    print(f"\nFrames {start}-{end}:")
    
    for cls in CLASSES:
        expected = int(row[cls])
        detected = log[(log['Class'] == cls) & (log['Frame'] >= start) & (log['Frame'] <= end) & (log['Direction'] == 'IN')]
        unique_ids = set(detected['Object ID'])
        found = len(unique_ids)
        
        # Debug: Show the actual IDs found in this interval
        if found > 0:
            print(f"  {cls}: expected {expected}, found {found} (IDs: {sorted(unique_ids)})")
        else:
            print(f"  {cls}: expected {expected}, found {found}")
            
        if found == expected:
            status = '✅ Correct'
            totals[cls]['correct'] += found
        elif found > expected:
            status = f'❌ Overcounted (+{found-expected})'
            totals[cls]['correct'] += expected
            totals[cls]['overcounted'] += (found-expected)
        else:
            status = f'❌ Missed ({expected-found})'
            totals[cls]['correct'] += found
            totals[cls]['missed'] += (expected-found)
        print(f"    {status}")

print("\n=== SUMMARY ===")
print(f"{'Class':<10} {'Correct':<8} {'Overcounted':<12} {'Missed':<8} {'Total':<8}")
for cls in CLASSES:
    total_calculated = totals[cls]['correct'] + totals[cls]['overcounted'] + totals[cls]['missed']
    print(f"{cls:<10} {totals[cls]['correct']:<8} {totals[cls]['overcounted']:<12} {totals[cls]['missed']:<8} {total_calculated:<8}")

print("\n=== VALIDATION ===")
print("Expected totals from ground truth:")
for cls in CLASSES:
    print(f"{cls}: {gt_totals[cls]}")

print("\nActual totals from log:")
for cls in CLASSES:
    print(f"{cls}: {log_totals[cls]}")

print("\nCalculated totals from interval analysis:")
for cls in CLASSES:
    calculated = totals[cls]['correct'] + totals[cls]['overcounted'] + totals[cls]['missed']
    print(f"{cls}: {calculated}")

print("\n=== POTENTIAL ISSUES ===")
for cls in CLASSES:
    gt_total = gt_totals[cls]
    log_total = log_totals[cls]
    calculated_total = totals[cls]['correct'] + totals[cls]['overcounted'] + totals[cls]['missed']
    
    if log_total != calculated_total:
        print(f"❌ {cls}: Log total ({log_total}) != Calculated total ({calculated_total})")
    if gt_total != calculated_total:
        print(f"❌ {cls}: Ground truth ({gt_total}) != Calculated total ({calculated_total})")
    if gt_total == log_total:
        print(f"✅ {cls}: Ground truth matches log total exactly")
    else:
        print(f"⚠️ {cls}: Ground truth ({gt_total}) != Log total ({log_total})")

print("Done.") 