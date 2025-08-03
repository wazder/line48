# Jupyter Terminal Komutları

## 1. Basit Test (Önerilen)

```bash
# Çalışma dizinine git
cd /workspace/line48

# Basit test scriptini çalıştır
python jupyter_test.py
```

## 2. Manuel Test

```bash
# Çalışma dizinine git
cd /workspace/line48

# Video kontrolü
ls -la videos/

# YOLO analizi
python src/run_analysis.py --video videos/short_video.mp4 --frame-logic --confidence 0.3 --iou 0.5 --imgsz 640

# SAM analizi
python src/run_sam_analysis.py --video videos/short_video.mp4 --sam-model vit_b --confidence 0.3 --iou 0.5 --imgsz 640 --download-sam

# Sonuçları kontrol et
ls -la logs/
ls -la outputs/
```

## 3. Hızlı Test

```bash
# Çalışma dizinine git
cd /workspace/line48

# Hızlı test scriptini çalıştır
python quick_line_test.py
```

## 4. Notebook Test

```bash
# Çalışma dizinine git
cd /workspace/line48

# Jupyter notebook'u başlat
jupyter notebook simple_line_test.ipynb
```

## 5. Sonuç Kontrolü

```bash
# Sonuç dosyalarını listele
ls -la logs/*.csv
ls -la outputs/*.mp4

# Sonuçları görüntüle
cat logs/*results*.csv
```

## 6. Performans Kontrolü

```bash
# GPU durumu
nvidia-smi

# Disk kullanımı
du -sh videos/ outputs/ logs/

# Bellek kullanımı
free -h
```

## 7. Temizlik

```bash
# Eski sonuçları temizle
rm -f logs/*.csv
rm -f outputs/*.mp4

# Cache temizle
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
```

## Önemli Notlar

1. **Video İndirme**: Eğer video yoksa önce indirin:
   ```bash
   pip install gdown
   gdown https://drive.google.com/uc?id=1rpgW_pmdMiUp_9_BVkRmXtF7Hon_u1gC -O videos/short_video.mp4
   ```

2. **SAM Bağımlılıkları**: SAM için gerekli paketler:
   ```bash
   pip install segment-anything matplotlib Pillow scipy
   ```

3. **Timeout Ayarları**: 
   - YOLO: 10 dakika
   - SAM: 20 dakika

4. **Çözünürlük**: 640x640 (hızlı test için)

5. **Confidence**: 0.3 (daha fazla tespit için)

## Beklenen Sonuçlar

- **YOLO**: Hızlı, ~2-5 dakika
- **SAM**: Yavaş, ~10-20 dakika
- **Karşılaştırma**: Benzer sonuçlar bekleniyor

## Hata Durumları

1. **Video bulunamadı**: `videos/short_video.mp4` dosyasını kontrol edin
2. **SAM hatası**: Bağımlılıkları yeniden yükleyin
3. **Timeout**: Daha küçük video kullanın veya timeout süresini artırın
4. **GPU hatası**: GPU memory'yi kontrol edin 