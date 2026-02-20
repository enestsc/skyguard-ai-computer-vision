import torch
from ultralytics import YOLO

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 1. Medium modeli çağırıyoruz
    model = YOLO('yolov8m.pt') 

    # 2. Daha derin öğrenme için epoch sayısını 50'de tutabiliriz 
    # ama 'patience' ekleyerek model gelişmiyorsa durmasını sağlayabiliriz.
    model.train(
        data='aeroplane-type-object-detection-v2/data.yaml', 
        epochs=50,
        imgsz=640,
        batch=16, # M4 için 16 güvenli bir limandır
        device=device,
        name='tasci_aviation_v2_medium'
    )

if __name__ == '__main__':
    main()