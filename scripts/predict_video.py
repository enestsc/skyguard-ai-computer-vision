import cv2
from ultralytics import YOLO
import torch

def main():
    # 1. Cihaz kontrolü (M4 ve MPS)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 2. SENİN EĞİTTİĞİN MODELİ YÜKLE
    model_path = 'runs/detect/tasci_aviation_v1/weights/best.pt'
    model = YOLO(model_path)
    
    # 3. TEST EDİLECEK VİDEO YOLU
    # İndirdiğin videonun ismini kontrol et (Örn: test_video.mp4)
    source_path = '../data/test_video.mp4' 
    
    # 4. TAHMİNİ DÖNGÜ İÇİNDE BAŞLAT
    # stream=True: Videoyu kare kare akıtır, belleği yormaz.
    results = model.predict(
        source=source_path,
        conf=0.25,        
        device=device,    
        save=True,        
        show=True,        # Ekranda pencere açar
        stream=True       # Akış modu
    )
    
    # EKRANIN KAPANMAMASI İÇİN DÖNGÜ
    for r in results:
        # Her kare işlendiğinde bu döngü çalışır.
        # 'q' tuşuna basarsan izlemeyi durdurur.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"İşlem tamamlandı. Sonuçlar 'runs/detect/predict' klasörüne kaydedildi.")

if __name__ == '__main__':
    main()