#!/usr/bin/env python3
"""Geliştirilmiş SAM Test - Yumuşatılmış Parametreler ve Büyük Model"""

import os
import subprocess
import time
import glob
import pandas as pd

def test_sam_improved():
    print("🔬 Geliştirilmiş SAM Testi")
    print("=" * 50)
    
    # Çalışma dizinini ayarla
    os.chdir('/workspace/line48')
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    # Video kontrolü
    video_path = "videos/short_video.mp4"
    if not os.path.exists(video_path):
        print("❌ Video bulunamadı!")
        return
    
    print(f"✅ Video bulundu: {video_path}")
    
    # Test 1: Yumuşatılmış parametrelerle vit_b
    print("\n🎯 Test 1: Yumuşatılmış parametrelerle vit_b")
    print("-" * 40)
    
    sam_cmd_1 = [
        "python", "src/run_sam_analysis.py",
        "--video", video_path,
        "--sam-model", "vit_b",
        "--confidence", "0.2",
        "--min-safe-time", "0.3",
        "--min-uncertain-time", "0.2", 
        "--min-very-brief-time", "0.1",
        "--imgsz", "640",
        "--download-sam"
    ]
    
    start_time = time.time()
    try:
        result_1 = subprocess.run(sam_cmd_1, capture_output=True, text=True, timeout=1800)
        time_1 = time.time() - start_time
        
        if result_1.returncode == 0:
            print(f"✅ Test 1 tamamlandı! Süre: {time_1:.1f}s")
        else:
            print(f"❌ Test 1 hatası: {result_1.stderr}")
    except Exception as e:
        print(f"❌ Test 1 hatası: {e}")
    
    # Test 2: Büyük model vit_l ile
    print("\n🎯 Test 2: Büyük model vit_l ile")
    print("-" * 40)
    
    sam_cmd_2 = [
        "python", "src/run_sam_analysis.py",
        "--video", video_path,
        "--sam-model", "vit_l",
        "--confidence", "0.2",
        "--min-safe-time", "0.3",
        "--min-uncertain-time", "0.2",
        "--min-very-brief-time", "0.1", 
        "--imgsz", "640",
        "--download-sam"
    ]
    
    start_time = time.time()
    try:
        result_2 = subprocess.run(sam_cmd_2, capture_output=True, text=True, timeout=3600)
        time_2 = time.time() - start_time
        
        if result_2.returncode == 0:
            print(f"✅ Test 2 tamamlandı! Süre: {time_2:.1f}s")
        else:
            print(f"❌ Test 2 hatası: {result_2.stderr}")
    except Exception as e:
        print(f"❌ Test 2 hatası: {e}")
    
    # Sonuçları karşılaştır
    print("\n📊 Sonuç Karşılaştırması:")
    print("=" * 40)
    
    try:
        # En son SAM sonuçlarını bul
        sam_files = glob.glob('logs/*sam*results*.csv')
        if sam_files:
            latest_sam = max(sam_files, key=os.path.getctime)
            print(f"📈 En son SAM sonuçları: {latest_sam}")
            
            df_sam = pd.read_csv(latest_sam)
            print("\n🔬 SAM Sonuçları:")
            print(df_sam)
            
            # Toplam line crossing sayısı
            total_crossings = df_sam['total_valid_crossings'].sum()
            print(f"\n📊 Toplam Line Crossing: {total_crossings}")
            
            if total_crossings > 0:
                print("✅ SAM line crossing tespit etti!")
            else:
                print("❌ SAM hala line crossing tespit etmiyor")
                
        else:
            print("❌ SAM sonuç dosyası bulunamadı")
            
    except Exception as e:
        print(f"❌ Sonuç analizi hatası: {e}")
    
    print(f"\n🎉 Test tamamlandı!")
    print(f"⏱️ Test 1 süresi: {time_1:.1f}s")
    print(f"⏱️ Test 2 süresi: {time_2:.1f}s")

if __name__ == "__main__":
    test_sam_improved() 