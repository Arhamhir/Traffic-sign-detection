# Traffic Sign & Light Detection

A simple computer vision project that detects traffic signs and traffic lights in images and videos using **YOLO11** (Ultralytics).

## What it detects

15 classes:

| Category | Classes |
|---|---|
| Traffic lights | Red Light, Green Light |
| Speed limits | 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 |
| Signs | Stop |

## Project files

| File | Purpose |
|---|---|
| `train.py` | Trains the model on the dataset |
| `predict.py` | Runs detection on `video.mp4` and shows the result |
| `data.yaml` | Dataset config (class names and train/val paths) |
| `traffic_model.pt` | Trained model weights (use this for predictions) |
| `yolo11m.pt`, `yolo11n.pt` | Base YOLO11 models used for training/starting point |
| `video.mp4` | Sample video to test on |
| `red_light.jpg`, `green_light.jpg`, `speed_limit.jpg` | Sample images |
| `runs/` | Training logs, charts and saved prediction outputs |

## Setup

```bash
pip install ultralytics opencv-python
```

> A CUDA-capable GPU is recommended for training, but prediction works on CPU too.

## Usage

### 1. Train the model

Update the `train` and `val` paths in `data.yaml` to point to your dataset, then run:

```bash
python train.py
```

This trains `yolo11m.pt` for 30 epochs at 640x640 image size. Results are saved in `runs/detect/`.

### 2. Run predictions

```bash
python predict.py
```

This loads `traffic_model.pt`, detects objects in `video.mp4`, displays the output, and saves it to `runs/detect/predict*/`.

To use your own image or video, change the `source` value in `predict.py`:

```python
model.predict(source='your_image.jpg', show=True, save=True)
```

## Dataset

The model was trained on the **Self-Driving Cars** dataset from Roboflow (4969 images, YOLO format, 416x416).

- Source: https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou
- License: CC BY 4.0

## Results

Trained for 30 epochs (640px, batch 32):

| Metric | Score |
|---|---|
| Precision | 0.925 |
| Recall | 0.963 |
| mAP@50 | 0.949 |
| mAP@50-95 | 0.559 |

Training curves and the confusion matrix are available in `runs/detect/train2/`.
