import pandas as pd

# Compare the two log files
old_log = 'logs/IMG_0015_blurred_log_20250727_192146.csv'  # Previous results
new_log = 'src/logs/IMG_0015_blurred_log_20250727_235225.csv'  # New optimized results

print("=== COMPARING OLD vs NEW RESULTS ===\n")

# Read both logs
old_df = pd.read_csv(old_log)
new_df = pd.read_csv(new_log)

print("OLD RESULTS (before optimization):")
old_totals = {}
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    class_detections = old_df[(old_df['Class'] == cls) & (old_df['Direction'] == 'IN')]
    unique_ids = set(class_detections['Object ID'])
    old_totals[cls] = len(unique_ids)
    print(f"  {cls}: {len(unique_ids)} unique objects")

print("\nNEW RESULTS (with optimization):")
new_totals = {}
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    class_detections = new_df[(new_df['Class'] == cls) & (new_df['Direction'] == 'IN')]
    unique_ids = set(class_detections['Object ID'])
    new_totals[cls] = len(unique_ids)
    print(f"  {cls}: {len(unique_ids)} unique objects")

print("\n=== COMPARISON ===")
print("Class      | Old | New | Difference")
print("-" * 35)
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    diff = new_totals[cls] - old_totals[cls]
    print(f"{cls:<10} | {old_totals[cls]:<3} | {new_totals[cls]:<3} | {diff:+d}")

print("\n=== CONFIDENCE DISTRIBUTION COMPARISON ===")
print("\nOLD LOG - Confidence distribution:")
old_conf_counts = old_df['Confidence'].value_counts()
for conf, count in old_conf_counts.items():
    print(f"  {conf}: {count}")

print("\nNEW LOG - Confidence distribution:")
new_conf_counts = new_df['Confidence'].value_counts()
for conf, count in new_conf_counts.items():
    print(f"  {conf}: {count}")

print("\n=== ANALYSIS ===")
if new_totals['handbag'] < old_totals['handbag']:
    print("❌ Handbag detection got WORSE with optimization")
    print("   This suggests the parameters were too aggressive")
elif new_totals['handbag'] > old_totals['handbag']:
    print("✅ Handbag detection IMPROVED with optimization")
else:
    print("➖ Handbag detection stayed the same")

print("\n=== RECOMMENDATIONS ===")
print("1. The optimization may have been too aggressive")
print("2. Try more conservative parameters:")
print("   - Confidence: 0.3 instead of 0.25")
print("   - NMS: 0.5 instead of 0.45")
print("   - Image size: 960 instead of 1280")
print("3. Or revert to original parameters and try different approach") 