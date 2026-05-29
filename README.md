# Transformer-Based Human Detection in Aerial Imagery

> Real-time bounding box prediction using RT-DETR and OpenCV on drone footage


## Overview

Detecting humans in aerial drone imagery is a challenging computer vision problem — objects appear as just a few pixels against complex backgrounds, and real-time performance is essential for operational use. This project implements and evaluates **RT-DETR** (Real-Time Detection Transformer) for human detection in aerial images, comparing it against YOLOv8 as a baseline. Real-time bounding box prediction is handled using **OpenCV**.

---

## Problem Statement

- **Small Object Detection** — humans at altitude occupy only a few pixels, making reliable localization difficult.
- **Speed vs. Accuracy Trade-off** — existing models tend to sacrifice one for the other.
- **Inconsistent Bounding Boxes** — poor localization causes unstable or overlapping boxes across frames.
- **Real-Time Requirement** — surveillance and rescue scenarios demand low-latency inference at scale.

---

## Tech Stack

| Component | Details |
|-----------|---------|
| **Dataset** | [HERIDAL](https://universe.roboflow.com/licenta-ynwvo/heridal-lrbkc) — aerial imagery annotated for human detection |
| **Primary Model** | RT-DETR (transformer-based object detection) |
| **Baseline Model** | YOLOv8 |
| **Framework** | [Ultralytics](https://docs.ultralytics.com/models/rtdetr/) |
| **Inference Tool** | OpenCV (video capture & frame-level inference) |

---

## Methodology

```
Collect Dataset → Preprocess Data → Train Model → Detect and Deploy
```

1. **Dataset** — HERIDAL aerial images captured from drones, annotated with human bounding boxes.
2. **Preprocessing** — images resized, split into train/test sets, and augmented for robustness.
3. **RT-DETR Model** — transformer-based detector using self-attention for global context.
4. **Feature Extraction** — self-attention layers capture spatial relationships across the full image.
5. **Bounding Box Prediction** — model outputs boxes, class labels, and confidence scores.
6. **Evaluation** — performance assessed with precision, recall, and mAP.
7. **Deployment** — real-time inference via OpenCV.

### How RT-DETR Works

- **Backbone** — scans the image at three zoom levels simultaneously for multi-scale features.
- **Hybrid Encoder** — merges zoom levels into a single rich representation; applies self-attention on wide-view features and fast convolutions elsewhere for real-time speed.
- **Decoder** — 300 learned object queries, each acting as a dedicated detector for one object.
- **Key Advantage** — no Non-Maximum Suppression (NMS) required. RT-DETR is trained to never produce overlapping predictions, which is critical for aerial imagery where NMS can accidentally suppress tiny humans.

---

## Results

### RT-DETR Metrics

| Metric | Score |
|--------|-------|
| mAP@50 | **0.731** |
| mAP@50–95 | 0.325 |
| Precision | **0.807** |
| Recall | 0.685 |

### RT-DETR vs YOLOv8

| Metric | RT-DETR | YOLOv8 |
|--------|---------|--------|
| mAP@50 | **0.73** | 0.53 |
| mAP@50–95 | **0.32** | 0.21 |
| Precision | **0.81** | 0.63 |
| Recall | **0.68** | 0.52 |

RT-DETR significantly outperforms YOLOv8, especially for small-object detection in aerial scenes.

---

## Discussion

**Why RT-DETR outperforms YOLOv8:**
- Self-attention captures global context across the entire image, not just local regions.
- Attention layers pick up fine-grained spatial detail that convolutional filters miss.
- No NMS overhead — cleaner detections in cluttered aerial scenes.

**Known Limitations:**
- mAP@50–95 of 0.325 indicates bounding box localization remains challenging, particularly in dense scenes and occlusions.

---

## Applications

- **Surveillance** — real-time crowd monitoring and perimeter security from drone feeds.
- **Search & Rescue** — rapid human localization in disaster zones and remote terrain.

---

## Future Work

- **Multi-Object Tracking** — extend detection to persistent ID tracking across frames.
- **Edge Deployment** — optimize and deploy on drone edge devices for on-board inference.
- **Improve Localization** — enhance bounding box regression to improve mAP@50–95.

---

## References

1. HERIDAL Dataset — Roboflow Universe: https://universe.roboflow.com/licenta-ynwvo/heridal-lrbkc
2. RT-DETR Model Documentation — Ultralytics: https://docs.ultralytics.com/models/rtdetr/
3. Ding, J., et al. (2021). Object detection in aerial images: A large-scale benchmark and challenges. *IEEE TPAMI*, 44(11), 7778–7796.
4. Caputo, S., et al. (2022). Human detection in drone images using YOLO for search-and-rescue operations. *AIxIA 2021*, LNCS vol. 13196. Springer. https://doi.org/10.1007/978-3-031-08421-8_22
