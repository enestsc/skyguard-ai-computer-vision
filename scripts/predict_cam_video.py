from ultralytics import YOLO
import torch
import cv2

# 1. Cihaz ve Model Hazırlığı
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = YOLO('runs/detect/tasci_aviation_v1/weights/best.pt')

# 2. Canlı Akış Döngüsü
# stream=True kullandığımız için sonuçları bir döngüde dönmeliyiz
results = model.predict(
    source="0",      # macOS bazen 0'ı string ("0") olarak bekleyebilir
    device=device, 
    show=True, 
    conf=0.3, 
    stream=True 
)

# Her bir kareyi (frame) işle ve ekranda tut
for r in results:
    # 'q' tuşuna basıldığında döngüden çık ve kamerayı kapat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()