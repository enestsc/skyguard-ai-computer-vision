import torch
from ultralytics import YOLO

def main():
    # 1. M4 İşlemcinin GPU gücünü (MPS) kontrol et
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"M4 GÜCÜ AKTİF: Eğitim {device.upper()} üzerinde başlıyor...")

    # 2. Hazır modeli temel alarak başla
    model = YOLO('yolov8n.pt') 

    # 3. EĞİTİMİ BAŞLAT
    # Path kısmına dikkat: scripts içinde olduğun için klasör adını doğru yazmalısın
    model.train(
        data='aeroplane-type-object-detection-1/data.yaml', 
        epochs=50,       # 50 tur tüm resimleri dön
        imgsz=640,       # Resim boyutu
        batch=16,        # M4 için 16 veya 32 idealdir
        device=device,   # Apple Silicon GPU (M4) kullanımı
        name='tasci_aviation_v1' 
    )

if __name__ == '__main__':
    main()