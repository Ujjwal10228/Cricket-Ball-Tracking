## Resulted Tracked Videos  
🔗 [View the tracked videos](https://drive.google.com/drive/folders/1QVbJsRWhZWkM5rUUoz-DSVXVF9HVbeYc?usp=sharing)


# Cricket Ball Detection \& Tracking in Broadcast Videos

This repository implements an **end-to-end cricket ball detection and tracking pipeline** designed for **realistic broadcast-style cricket videos**.  
The system combines a **YOLOv8-based object detector** with a **motion-gated Kalman filter** to produce **robust ball trajectories and frame-wise annotations**, while explicitly avoiding hallucinated results.

---

## 1. Project Objectives

The primary goals of this project are:

- Detect the cricket ball in video frames where it is **visually separable**
- Track the ball across frames using **temporal filtering**
- Generate:
  - Frame-wise CSV annotations:  
    \[
    (\text{frame}, x, y, \text{visible})
    \]
  - Processed videos with centroid and trajectory overlays
- Handle real-world challenges such as:
  - Motion blur during delivery
  - Small object size
  - Occlusions
  - False positives (helmets, caps, background clutter)

The system is intentionally **conservative**:

\[
\text{If sufficient evidence is not available, tracking is disabled.}
\]

---

**Requirements**

pip install -r requirements.txt

**Batch Inference**

From the project root:

python code/main.py \
  --input test_videos \
  --output results \
  --annotations annotations \
  --model runs/train/cricket_ball/weights/best.pt

## 2. Repository Structure

```text
project_root/
│
├── code/
│   ├── main.py          # Entry point (batch inference)
│   ├── inference.py     # YOLOv8 inference wrapper
│   ├── kalman.py        # Constant-velocity Kalman filter
│   ├── tracker.py       # Motion-gated tracking logic
│   ├── utils.py         # Visualization utilities
│
├── test_videos/         # Input videos
│
├── annotations/         # CSV outputs
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
│
├── results/             # Tracked output videos link is provided
│   ├── 1_tracked.mp4
│   ├── 2_tracked.mp4
│   └── ...
│
├── runs/train/cricket_ball
|    |── weights
|        |── best.pt             # Trained YOLOv8 model
├── requirements.txt     # Dependencies
├── report.pdf           # Detailed technical report
├── README.md


