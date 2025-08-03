#!/usr/bin/env python3
"""Hızlı Line Test - YOLO vs SAM Karşılaştırması"""

import os
import subprocess
import time
import glob
import pandas as pd

def main():
    print("🚀 Hızlı Line Test Başlatılıyor...")
    print("=" * 50)
    
    # Çalışma dizinini ayarla
    os.chdir('/workspace/line48')
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    # Gerekli dizinleri oluştur
    os.makedirs('videos', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Video kontrolü
    video_path = "videos/short_video.mp4"
    if not os.path.exists(video_path):
        print("❌ Video bulunamadı! Lütfen önce video indirin.")
        return
    
    print(f"✅ Video bulundu: {video_path}")
    
    # YOLO Analizi
    print("\n🎯 YOLO Analizi Başlatılıyor...")
    yolo_start = time.time()
    
    yolo_cmd = [
        "python", "src/run_analysis.py",
        "--video", video_path,
        "--frame-logic",
        "--confidence", "0.3",
        "--iou", "0.5", 
        "--imgsz", "640"
    ]
    
    try:
        yolo_result = subprocess.run(yolo_cmd, capture_output=True, text=True, timeout=600)
        yolo_time = time.time() - yolo_start
        
        if yolo_result.returncode == 0:
            print(f"✅ YOLO tamamlandı! Süre: {yolo_time:.1f}s")
        else:
            print(f"❌ YOLO hatası: {yolo_result.stderr}")
            return
    except subprocess.TimeoutExpired:
        print("⏰ YOLO zaman aşımı")
        return
    except Exception as e:
        print(f"❌ YOLO hatası: {e}")
        return
    
    # SAM Analizi
    print("\n🔬 SAM Analizi Başlatılıyor...")
    sam_start = time.time()
    
    sam_cmd = [
        "python", "src/run_sam_analysis.py",
        "--video", video_path,
        "--sam-model", "vit_b",
        "--confidence", "0.3",
        "--iou", "0.5",
        "--imgsz", "640",
        "--download-sam"
    ]
    
    try:
        sam_result = subprocess.run(sam_cmd, capture_output=True, text=True, timeout=1200)
        sam_time = time.time() - sam_start
        
        if sam_result.returncode == 0:
            print(f"✅ SAM tamamlandı! Süre: {sam_time:.1f}s")
        else:
            print(f"❌ SAM hatası: {sam_result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ SAM zaman aşımı")
    except Exception as e:
        print(f"❌ SAM hatası: {e}")
    
    # Sonuçları karşılaştır
    print("\n📊 Sonuç Karşılaştırması:")
    print("=" * 40)
    
    try:
        # YOLO sonuçları
        yolo_files = glob.glob('logs/*results*.csv')
        yolo_files = [f for f in yolo_files if 'sam' not in f]
        
        # SAM sonuçları
        sam_files = glob.glob('logs/*sam_results*.csv')
        
        if yolo_files and sam_files:
            latest_yolo = max(yolo_files, key=os.path.getctime)
            latest_sam = max(sam_files, key=os.path.getctime)
            
            df_yolo = pd.read_csv(latest_yolo)
            df_sam = pd.read_csv(latest_sam)
            
            print(f"{'Sınıf':<12} {'YOLO':<8} {'SAM':<8} {'Fark':<8}")
            print("-" * 40)
            
            total_yolo = 0
            total_sam = 0
            
            for _, row in df_yolo.iterrows():
                cls = row.get('class', 'unknown')
                yolo_total = row.get('total_valid_crossings', 0)
                
                sam_row = df_sam[df_sam['class'] == cls]
                sam_total = sam_row['total_valid_crossings'].iloc[0] if len(sam_row) > 0 else 0
                
                total_yolo += yolo_total
                total_sam += sam_total
                
                diff = sam_total - yolo_total
                print(f"{cls:<12} {yolo_total:<8} {sam_total:<8} {diff:+<8}")
            
            print("-" * 40)
            overall_diff = total_sam - total_yolo
            print(f"{'TOPLAM':<12} {total_yolo:<8} {total_sam:<8} {overall_diff:+<8}")
            
            # Performans
            print(f"\n⚡ Performans:")
            print(f"YOLO: {yolo_time:.1f}s")
            print(f"SAM:  {sam_time:.1f}s")
            speed_ratio = yolo_time / sam_time if sam_time > 0 else 0
            print(f"Hız oranı: {1/speed_ratio:.1f}x")
            
            # Sonuç
            print(f"\n🏆 Sonuç:")
            if abs(overall_diff) < 5:
                print("✅ SAM ve YOLO benzer sonuçlar!")
            elif overall_diff > 0:
                print("✅ SAM daha iyi!")
            else:
                print("🔴 YOLO daha iyi!")
                
        else:
            print("❌ Sonuç dosyaları bulunamadı")
            
    except Exception as e:
        print(f"❌ Karşılaştırma hatası: {e}")
    
    print(f"\n🎉 Test tamamlandı!")
    print(f"⏱️ Toplam süre: {yolo_time + sam_time:.1f}s")

if __name__ == "__main__":
    main() 