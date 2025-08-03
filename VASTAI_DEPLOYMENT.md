# VastAI'da LineLogic Deployment Rehberi

Bu rehber LineLogic projesini VastAI platformunda çalıştırmak için gerekli adımları içerir.

## 1. VastAI Instance Gereksinimleri

### Minimum Sistem Gereksinimleri:
- **GPU**: NVIDIA GTX 1660 Ti veya üzeri (6GB+ VRAM)
- **RAM**: 16GB sistem RAM
- **Disk**: 50GB+ depolama alanı
- **CUDA**: 12.1+ destekli

### Önerilen Konfigürasyon:
- **GPU**: RTX 3080/4080 veya A100 (24GB+ VRAM)
- **RAM**: 32GB+ sistem RAM
- **Disk**: 100GB+ NVMe SSD

## 2. VastAI Instance Kurulumu

### Adım 1: Instance Seçimi
```bash
# VastAI CLI ile uygun instance ara
vastai search offers 'cuda_vers >= 12.1 gpu_ram >= 6 ram >= 16'

# Instance kiraları (örnek)
vastai create instance <instance_id> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

### Adım 2: Proje Klonlama
```bash
# Instance'a bağlan
ssh root@<instance_ip>

# Proje dizinini oluştur
cd /workspace
git clone <your-repo-url> line48
cd line48
```

## 3. Kurulum Seçenekleri

### Seçenek A: Docker ile Kurulum
```bash
# Docker image'ı build et
docker build -t linelogic .

# Container'ı çalıştır
docker run --gpus all -it --rm \
  -v /workspace/line48/videos:/workspace/line48/videos \
  -v /workspace/line48/outputs:/workspace/line48/outputs \
  -v /workspace/line48/logs:/workspace/line48/logs \
  linelogic
```

### Seçenek B: Manuel Kurulum
```bash
# Setup scriptini çalıştır
chmod +x vastai_setup.sh
./vastai_setup.sh

# Sanal ortamı aktive et
source /workspace/line48_env/bin/activate
```

## 4. Video Yükleme

### Videos Dizinine Yükleme:
```bash
# SCP ile local'den yükle
scp your_video.mp4 root@<instance_ip>:/workspace/line48/videos/

# Wget ile URL'den indir
cd /workspace/line48/videos
wget <video_url>

# Jupyter notebook üzerinden upload (eğer varsa)
```

## 5. Analiz Çalıştırma

### Temel Kullanım:
```bash
cd /workspace/line48/src
python run_analysis.py --list-videos
python run_analysis.py --video "../videos/your_video.mp4" --frame-logic
```

### Parametreli Çalıştırma:
```bash
# Yüksek hassasiyet için
python run_analysis.py \
  --video "../videos/your_video.mp4" \
  --confidence 0.3 \
  --iou 0.4 \
  --imgsz 1280 \
  --frame-logic
```

### Batch İşleme:
```bash
# Tüm videoları işle
for video in ../videos/*.mp4; do
    python run_analysis.py --video "$video" --frame-logic
done
```

## 6. Sonuçları İndirme

### Output Dosyalarını Local'e Çek:
```bash
# Processed videos
scp -r root@<instance_ip>:/workspace/line48/outputs/ ./

# Log dosyaları
scp -r root@<instance_ip>:/workspace/line48/logs/ ./
```

## 7. Performance İpuçları

### GPU Optimizasyonu:
- `--imgsz 1280` büyük videolar için
- `--imgsz 640` hızlı işleme için
- Batch size artırımı için VRAM'i izle

### Memory Yönetimi:
```python
# Büyük videolar için memory temizligi
import torch
torch.cuda.empty_cache()
```

## 8. Sorun Giderme

### CUDA Hatası:
```bash
# CUDA versiyonunu kontrol et
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### FFmpeg Hatası:
```bash
# FFmpeg'i yeniden yükle
apt-get update && apt-get install -y ffmpeg
```

### Memory Yetersizliği:
```bash
# Swap alanını artır
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
```

## 9. Maliyet Optimizasyonu

### Instance Yönetimi:
- İşlem bittiğinde instance'ı durdur
- Spot instance'ları kullan (daha ucuz)
- Gerekenden fazla GPU kiralam

### Veri Transferi:
- Büyük videoları VastAI storage'a yükle
- Compressed format kullan
- Batch processing yap

## 10. Örnek Workflow

```bash
# 1. Instance başlat
vastai create instance <id> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Kurulum
ssh root@<ip>
cd /workspace && git clone <repo> line48 && cd line48
./vastai_setup.sh && source /workspace/line48_env/bin/activate

# 3. Video yükle
scp video.mp4 root@<ip>:/workspace/line48/videos/

# 4. Analizi çalıştır
cd /workspace/line48/src
python run_analysis.py --video "../videos/video.mp4" --frame-logic

# 5. Sonuçları indir
scp -r root@<ip>:/workspace/line48/outputs/ ./

# 6. Instance'ı durdur
vastai destroy instance <id>
```

Bu rehber VastAI'da LineLogic projesini başarıyla çalıştırmanız için gerekli tüm adımları içerir.