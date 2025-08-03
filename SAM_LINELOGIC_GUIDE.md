# SAM + LineLogic Kullanım Rehberi

Bu rehber Segment Anything Model (SAM) ile LineLogic'in entegre kullanımını açıklar.

## 🎯 SAM + LineLogic Avantajları

### Geleneksel YOLO vs SAM Hybrid:
- **YOLO**: Hızlı bbox detection, yaklaşık sınırlar
- **SAM**: Piksel-perfect segmentasyon, hassas sınırlar
- **Hybrid**: YOLO'nun hızı + SAM'ın hassasiyeti

### Neden SAM Kullanmalı?
1. **Piksel-level hassasiyet** - Çok daha kesin nesne sınırları
2. **Daha iyi tracking** - Segment centroids daha kararlı
3. **Gelişmiş line crossing** - Mask ile çizgi kesişimi daha doğru
4. **Visual quality** - Çok daha güzel görselleştirme

## 📦 Kurulum

### VastAI'da Hızlı Kurulum:
```bash
cd /workspace
git clone https://github.com/wazder/line48.git
cd line48
chmod +x vastai_sam_setup.sh
./vastai_sam_setup.sh
```

### Manuel Kurulum:
```bash
pip install segment-anything ultralytics supervision opencv-python numpy matplotlib Pillow scipy pandas

# SAM model indir (375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 🚀 Kullanım

### Temel SAM Analizi:
```bash
source /workspace/sam_line48_env/bin/activate
cd /workspace/line48/src

python run_sam_analysis.py \
  --video "../videos/input_video.mp4" \
  --sam-model vit_b
```

### RTX 4090 için Optimize:
```bash
# Yüksek kalite (ViT-H model)
python run_sam_analysis.py \
  --video "../videos/input_video.mp4" \
  --sam-model vit_h \
  --imgsz 1280 \
  --confidence 0.25

# Hızlı işleme (frame skipping)
python run_sam_analysis.py \
  --video "../videos/input_video.mp4" \
  --sam-model vit_b \
  --skip-frames 2 \
  --imgsz 1024
```

### Google Drive Video ile:
```bash
# Video indir
gdown "https://drive.google.com/uc?id=1OYdAf3OMYIFLnGAi8Gx9an_xv3BulPpN" -O videos/input_video.mp4

# SAM analizi çalıştır
python run_sam_analysis.py \
  --video "../videos/input_video.mp4" \
  --sam-model vit_b \
  --save-video \
  --verbose
```

## ⚙️ SAM Model Seçimi

| Model | Boyut | Hız | Kalite | RTX 4090 Önerisi |
|-------|-------|-----|--------|-------------------|
| vit_b | 375MB | Hızlı | İyi | ✅ Günlük kullanım |
| vit_l | 1.25GB | Orta | Çok İyi | ✅ Yüksek kalite |
| vit_h | 2.56GB | Yavaş | En İyi | ✅ En iyi sonuç |

### Model Performansı (RTX 4090):
- **vit_b**: ~15-25 FPS
- **vit_l**: ~8-15 FPS  
- **vit_h**: ~4-10 FPS

## 📊 Parametre Rehberi

### YOLO Parametreleri:
```bash
--confidence 0.25      # Nesne tespit eşiği
--iou 0.45            # NMS IoU eşiği
--imgsz 1280          # Girdi görüntü boyutu
```

### SAM Parametreleri:
```bash
--sam-model vit_b     # SAM model tipi
--sam-checkpoint path # Model dosya yolu
--download-sam        # Model otomatik indir
```

### Frame Logic Parametreleri:
```bash
--min-safe-time 0.5        # Güvenli tespit süresi
--min-uncertain-time 0.28  # Belirsiz tespit süresi  
--min-very-brief-time 0.17 # Çok kısa tespit süresi
```

### İşleme Parametreleri:
```bash
--max-frames 1000     # Test için maksimum frame
--skip-frames 2       # Her N frame'i işle
--device cuda:0       # GPU device
```

## 🎬 Çıktı Dosyaları

### Video Çıktısı:
- `outputs/video_sam_processed_timestamp.mp4`
- Segmentasyon maskleri + çizgiler + sayımlar

### Log Dosyaları:
- `logs/video_sam_log_timestamp.csv` - Detaylı tespit logu
- `logs/video_sam_results_timestamp.csv` - Özet sonuçlar

### Log İçeriği:
```csv
frame_idx, timestamp, class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_area, mask_score
```

## 🔧 Performans Optimizasyonu

### RTX 4090 için En İyi Ayarlar:
```bash
# Kalite odaklı
python run_sam_analysis.py \
  --video video.mp4 \
  --sam-model vit_h \
  --imgsz 1280 \
  --confidence 0.2 \
  --device cuda:0

# Hız odaklı  
python run_sam_analysis.py \
  --video video.mp4 \
  --sam-model vit_b \
  --imgsz 1024 \
  --skip-frames 2 \
  --confidence 0.3
```

### Memory Optimizasyonu:
```bash
# Büyük videolar için
python run_sam_analysis.py \
  --video video.mp4 \
  --sam-model vit_b \
  --imgsz 640 \
  --skip-frames 3 \
  --max-frames 5000
```

## 📈 Sonuç Analizi

### SAM vs YOLO Karşılaştırması:
```bash
# YOLO analizi
python run_analysis.py --video video.mp4 --frame-logic

# SAM analizi  
python run_sam_analysis.py --video video.mp4 --sam-model vit_b

# Sonuçları karşılaştır
python analysis_tools/compare_sam_yolo.py
```

### Beklenen İyileştirmeler:
- **Hassasiyet**: %10-15 daha iyi
- **False positives**: %20-30 azalma
- **Tracking kararlılığı**: Belirgin iyileşme
- **Görsel kalite**: Çok daha iyi

## 🚨 Sorun Giderme

### SAM Model Bulunamadı:
```bash
# Modeli manuel indir
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Veya otomatik indir
python run_sam_analysis.py --download-sam --sam-model vit_b
```

### CUDA Memory Hatası:
```bash
# Daha küçük model kullan
--sam-model vit_b --imgsz 640

# Frame skipping
--skip-frames 4

# Max frames sınırla
--max-frames 2000
```

### Yavaş İşleme:
```bash
# Frame skipping kullan
--skip-frames 2

# Küçük image size
--imgsz 1024

# Hızlı SAM model
--sam-model vit_b
```

## 🎯 En İyi Pratikler

### 1. Model Seçimi:
- **Test/Hızlı**: vit_b
- **Production**: vit_l
- **En İyi Kalite**: vit_h

### 2. Parametre Tuning:
- Yüksek confidence → Az false positive
- Düşük confidence → Daha fazla tespit
- Büyük imgsz → Daha iyi kalite, yavaş
- Frame skipping → Hız artışı

### 3. Video Preprocessing:
- 1080p optimal çözünürlük
- Sabit FPS tercih et
- Kaliteli video kullan

### 4. VastAI Kullanımı:
- RTX 4090/A100 tercih et
- Yeterli VRAM'e dikkat et
- Batch processing yap

Bu rehber SAM + LineLogic'in tam potansiyelini kullanmanızı sağlar!