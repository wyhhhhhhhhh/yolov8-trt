from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-obb.yaml").load('yolov8n-obb.pt')  # build a new model from YAML

# Train the model
results = model.train(data="datasets/real_data.yaml", epochs=200, imgsz=(480, 640), task = 'obb', device=0, workers=4, batch=8)