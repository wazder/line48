#!/usr/bin/env python3
"""Check the actual SAM results file"""

import os
import glob
import pandas as pd

def check_results():
    print("🔍 Checking SAM Results Files")
    print("=" * 40)
    
    # Find all SAM results files
    sam_files = glob.glob('logs/*sam*results*.csv')
    
    if not sam_files:
        print("❌ No SAM results files found")
        return
    
    print(f"📁 Found {len(sam_files)} SAM results files:")
    for file in sam_files:
        print(f"   - {file}")
    
    # Get the most recent file
    latest_file = max(sam_files, key=os.path.getctime)
    print(f"\n📈 Most recent file: {latest_file}")
    
    # Read and display the results
    try:
        df = pd.read_csv(latest_file)
        print(f"\n📊 Results from {latest_file}:")
        print(df)
        
        # Check for crossings
        if 'total_valid_crossings' in df.columns:
            total_crossings = df['total_valid_crossings'].sum()
            print(f"\n🎯 Total valid crossings: {total_crossings}")
            
            if total_crossings > 0:
                print("✅ SUCCESS! Line crossings detected!")
            else:
                print("❌ No line crossings detected")
        else:
            print("⚠️ No 'total_valid_crossings' column found")
            print("Available columns:", df.columns.tolist())
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    check_results() 