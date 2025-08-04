#!/usr/bin/env python3
"""Fixed SAM Test - With Relaxed Parameters"""

import os
import subprocess
import time
import glob
import pandas as pd

def test_sam_fixed():
    print("🔧 Fixed SAM Testi")
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
    
    # Test: Çok yumuşatılmış parametrelerle
    print("\n🎯 Test: Çok yumuşatılmış parametrelerle")
    print("-" * 40)
    
    sam_cmd = [
        "python", "src/run_sam_analysis.py",
        "--video", video_path,
        "--sam-model", "vit_b",
        "--confidence", "0.01",  # Çok düşük
        "--min-safe-time", "0.01",  # Çok kısa
        "--min-uncertain-time", "0.005",
        "--min-very-brief-time", "0.001",
        "--imgsz", "640",
        "--max-frames", "20",  # Sadece ilk 20 frame
        "--verbose"
    ]
    
    print("Running command:", " ".join(sam_cmd))
    
    start_time = time.time()
    try:
        result = subprocess.run(sam_cmd, capture_output=True, text=True, timeout=1800)
        time_taken = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Test tamamlandı! Süre: {time_taken:.1f}s")
            print("STDOUT:")
            print(result.stdout[-3000:])  # Son 3000 karakter
        else:
            print(f"❌ Test hatası: {result.stderr}")
    except Exception as e:
        print(f"❌ Test hatası: {e}")
    
    # Sonuçları kontrol et
    print("\n📊 Sonuç Kontrolü:")
    print("=" * 40)
    
    try:
        # En son SAM sonuçlarını bul - look for the most recent file
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
                print(f"🎉 Başarılı! {total_crossings} line crossing tespit edildi!")
            else:
                print("❌ SAM hala line crossing tespit etmiyor")
                
        else:
            print("❌ SAM sonuç dosyası bulunamadı")
            
    except Exception as e:
        print(f"❌ Sonuç analizi hatası: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Test tamamlandı!")
    print(f"⏱️ Test süresi: {time_taken:.1f}s")

if __name__ == "__main__":
    test_sam_fixed() 