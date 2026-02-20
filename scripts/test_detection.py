from ultralytics import YOLO

# Hazır modeli yükle
model = YOLO('yolov8n.pt')

# Bir klasör dolusu resmi tahmin et
results = model.predict(source='../data/test_images', save=True, conf=0.5)