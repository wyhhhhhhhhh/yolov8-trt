from ultralytics import YOLO
model = YOLO("best.pt")
success = model.export(format="onnx", opset=11, simplify=True, half=True)  # export the model to onnx format
assert success

# from ultralytics import YOLO

# model = YOLO("best.pt")  # load a pretrained model (recommended for training)
# success = model.export(format="engine", device=0)  # export the model to engine format
# assert success