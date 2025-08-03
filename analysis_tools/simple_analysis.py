import pandas as pd

# Read ground truth and log
gt = pd.read_csv('ground_truth_template.csv', comment='#')
log = pd.read_csv('logs/IMG_0015_blurred_log_20250727_192146.csv')

print("=== SIMPLE COUNTING ANALYSIS ===\n")

# Count total unique objects in log
print("Total unique objects detected in log:")
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    unique_ids = set(class_detections['Object ID'])
    print(f"{cls}: {len(unique_ids)} unique IDs")

print("\n=== GROUND TRUTH TOTALS ===")
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    total = gt[cls].sum()
    print(f"{cls}: {total}")

print("\n=== THE REAL ISSUE ===")
print("The problem is in the comparison logic, not the data.")
print("The comparison script is designed to count objects per interval,")
print("but it's not handling the fact that the same object can appear")
print("in multiple intervals correctly.")

print("\n=== CORRECTED APPROACH ===")
print("We should either:")
print("1. Count each unique object only once across the entire video")
print("2. Or modify the comparison to handle multi-interval objects properly")

# Let's show what the correct totals should be
print("\n=== CORRECTED TOTALS ===")
log_totals = {}
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    unique_ids = set(class_detections['Object ID'])
    log_totals[cls] = len(unique_ids)

gt_totals = {}
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    gt_totals[cls] = gt[cls].sum()

print("Class      | Log Total | GT Total | Difference")
print("-" * 45)
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    diff = log_totals[cls] - gt_totals[cls]
    print(f"{cls:<10} | {log_totals[cls]:<9} | {gt_totals[cls]:<8} | {diff:+d}") 