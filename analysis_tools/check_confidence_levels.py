import pandas as pd

# Read the log
log = pd.read_csv('logs/IMG_0015_blurred_log_20250727_192146.csv')

print("=== CONFIDENCE LEVELS IN LOG ===\n")

# Check all confidence levels
print("All confidence levels in log:")
confidence_counts = log['Confidence'].value_counts()
for confidence, count in confidence_counts.items():
    print(f"  {confidence}: {count}")

print("\n=== CONFIDENCE BY CLASS ===")
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    print(f"\n{cls.capitalize()}:")
    confidence_counts = class_detections['Confidence'].value_counts()
    for confidence, count in confidence_counts.items():
        print(f"  {confidence}: {count}")

print("\n=== TOTAL UNIQUE OBJECTS BY CLASS ===")
for cls in ['backpack', 'person', 'handbag', 'suitcase']:
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    unique_ids = set(class_detections['Object ID'])
    print(f"{cls}: {len(unique_ids)} unique objects")

print("\n=== CONCLUSION ===")
print("The comparison script counts ALL detections regardless of confidence level.")
print("So 'very_brief' and 'uncertain' detections are already included.")
print("The real issue is that the model is not detecting enough handbags/backpacks.") 