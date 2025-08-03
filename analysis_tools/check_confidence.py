import pandas as pd

# Read the log
log = pd.read_csv('logs/IMG_0015_blurred_log_20250727_192146.csv')

print("=== CONFIDENCE LEVEL ANALYSIS ===\n")

# Check what confidence levels exist
print("All confidence levels in the log:")
confidence_levels = log['Confidence'].value_counts()
for level, count in confidence_levels.items():
    print(f"  {level}: {count}")

print("\n=== BY CLASS ===")
for cls in ['handbag', 'backpack', 'person', 'suitcase']:
    class_detections = log[(log['Class'] == cls) & (log['Direction'] == 'IN')]
    print(f"\n{cls.capitalize()}:")
    confidence_counts = class_detections['Confidence'].value_counts()
    for confidence, count in confidence_counts.items():
        print(f"  {confidence}: {count}")

print("\n=== WHAT'S BEING COUNTED ===")
print("Based on the frame logic, these confidence levels are counted:")
print("- 'safe': ✅ Counted")
print("- 'uncertain': ✅ Counted") 
print("- 'very_brief': ❌ NOT counted (discarded)")
print("- 'discard': ❌ NOT counted")

print("\n=== VERY BRIEF DETECTIONS (NOT COUNTED) ===")
very_brief = log[log['Confidence'] == 'very_brief']
print(f"Total very_brief detections: {len(very_brief)}")
for _, row in very_brief.iterrows():
    print(f"  Object {row['Object ID']} ({row['Class']}) at frame {row['Frame']}")

print("\n=== UNCERTAIN DETECTIONS (COUNTED) ===")
uncertain = log[log['Confidence'] == 'uncertain']
print(f"Total uncertain detections: {len(uncertain)}")
for _, row in uncertain.iterrows():
    print(f"  Object {row['Object ID']} ({row['Class']}) at frame {row['Frame']}")

print("\n=== CONCLUSION ===")
print("'very_brief' detections are NOT being counted in the current results.")
print("Only 'safe' and 'uncertain' detections are counted.")
print("Counting 'very_brief' would improve accuracy.") 