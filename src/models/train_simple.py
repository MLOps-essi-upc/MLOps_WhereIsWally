from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='../../data/raw/yolov8_format/data.yaml', epochs=2, imgsz=300)