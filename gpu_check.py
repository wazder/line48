#!/usr/bin/env python3
"""GPU Kullanım Kontrolü"""

import torch
import subprocess
import os

def check_gpu():
    print("🎮 GPU Kontrolü")
    print("=" * 30)
    
    # PyTorch GPU kontrolü
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔄 CUDA Version: {torch.version.cuda}")
        
        # GPU memory durumu
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("❌ CUDA kullanılamıyor - CPU kullanılıyor")
    
    # nvidia-smi kontrolü
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv'], 
                              capture_output=True, text=True)
        print(f"\n📊 nvidia-smi çıktısı:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ nvidia-smi hatası: {e}")
    
    # SAM GPU testi
    print(f"\n🔬 SAM GPU Testi:")
    try:
        from src.sam_utils import SAMLineLogic
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"📱 SAM device: {device}")
        
        # SAM'i test et
        sam_logic = SAMLineLogic(device=device)
        print("✅ SAM GPU'da başarıyla yüklendi")
        
        # Test frame oluştur
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        segmented_frame, detections = sam_logic.detect_and_segment(test_frame)
        print(f"✅ SAM test başarılı - {len(detections)} detection")
        
    except Exception as e:
        print(f"❌ SAM GPU test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_gpu() 