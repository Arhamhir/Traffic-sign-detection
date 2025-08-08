from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(data = "data.yaml",imgsz = 640,batch = 32,epochs = 30, workers = 0, device = 0)
