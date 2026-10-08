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

Trained for 30 epochs (640px):

| Metric | Score |
|---|---|
| Precision | 0.925 |
| Recall | 0.963 |
| mAP@50 | 0.949 |
| mAP@50-95 | 0.559 |

Training curves and the confusion matrix are available in `runs/detect/train2/`.

---

## Technical details

### How detection works

`predict.py` runs the standard single-stage YOLO pipeline:

1. **Preprocess** — the frame is resized to 640x640 (letterboxed to keep aspect ratio) and normalized to 0-1.
2. **Backbone** — a convolutional stack (C3k2 + C2PSA attention blocks) extracts multi-scale feature maps at 1/8, 1/16 and 1/32 resolution.
3. **Neck (PAN-FPN)** — top-down and bottom-up feature fusion so small signs and large signs are both visible to the head.
4. **Head** — a decoupled anchor-free head predicts, for every grid cell: a class score, a box (as a distance to the four sides), and a box distribution.
5. **Post-process** — score thresholding (conf ≈ 0.25), Non-Maximum Suppression at IoU 0.7, keeping at most 300 boxes per frame.

Because the model outputs raw distributions rather than a single offset, the final box coordinates come from a **Distribution Focal Loss (DFL)** regression — this tends to give tighter boxes than plain L2 regression.

### Training setup

| Setting | Value | Why it matters |
|---|---|---|
| Base model | `yolo11m.pt` | Medium variant — better accuracy than `yolo11n`, slower than `yolo11n` |
| Weights | COCO-pretrained, fine-tuned | Transfer learning: edges/textures already learned, so 30 epochs is enough |
| Input size | 640x640 | Dataset ships at 416x416, so images are upscaled — limits recall on tiny signs |
| Epochs / batch | 30 / 16 | ~4.5k images, converged well before epoch 30 |
| Optimizer | SGD (auto), lr 0.01, momentum 0.937, wd 5e-4 | Standard YOLO recipe, with 3-epoch linear warmup |
| Precision | AMP (mixed fp16/fp16-fp32) | Halves VRAM, speeds up training on GPU |

**Loss** = weighted sum of three terms:

- `box` 7.5 — Complete IoU between predicted and ground-truth box
- `cls` 0.5 — binary cross-entropy per class
- `dfl` 1.5 — cross-entropy over the box distance distribution

The heavy `box` weighting explains the high mAP@50 but lower mAP@50-95: boxes are usually in the right place, but not pixel-perfect.

**Augmentation** (applied on the fly): Mosaic (4-image collage, disabled for the last 10 epochs to sharpen final boxes), HSV jitter, 10% translate, 50% scale jitter, horizontal flip, and random erasing.

### Reading the metrics

- **Precision 0.925 / Recall 0.963** — very few false positives *and* few missed signs; the detector is trustworthy for this class set.
- **mAP@50 0.949** — nearly all objects are detected with IoU ≥ 0.5.
- **mAP@50-95 0.559** — the drop is expected: it averages over much stricter IoU thresholds (0.5 → 0.95) and is heavily penalized by small objects and the 416→640 upscale. Improving it means training at higher resolution with more epochs, not changing the architecture.

### Things to know before adapting it

- `data.yaml` currently points at **absolute Windows paths** (`D:\me\...`) — you must edit `train` and `val` before training elsewhere.
- `train.py` says `batch=32`, but the saved run in `runs/detect/train2/` actually used `batch=16`. Check `runs/detect/train*/args.yaml` for the true configuration of any run.
- Two training runs exist: `train/` (early/aborted) and `train2/` (the complete 30-epoch run whose weights became `traffic_model.pt`).
- Class imbalance: speed-limit classes dominate the label distribution (see `runs/detect/train2/labels.jpg`), so traffic lights are the classes most likely to be confused.
- To deploy, export with `model.export(format="onnx")` for a framework-agnostic, faster inference path.
