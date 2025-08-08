from ultralytics import YOLO

model = YOLO("traffic_model.pt")
model.predict(source='video.mp4',show=True, save=True)