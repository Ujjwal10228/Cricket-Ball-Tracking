import cv2
import csv
import os
from inference import BallDetector
from tracker import BallTracker
from utils import draw_centroid, draw_trajectory

# ---------------- CONFIG ----------------
TEST_VIDEO_DIR = "test_videos"
MODEL_PATH = "runs/train/cricket_ball/weights/best.pt"

ANNOTATION_DIR = "annotations"
RESULT_DIR = "results"

os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
# ----------------------------------------


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"▶ Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_video_path = os.path.join(RESULT_DIR, f"{video_name}_tracked.mp4")
    csv_path = os.path.join(ANNOTATION_DIR, f"{video_name}.csv")

    out = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    detector = BallDetector(MODEL_PATH)
    tracker = BallTracker(fps, H)

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "x", "y", "visible"])

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame)
        center, visible = tracker.step(detection)

        if center is not None:
            draw_centroid(frame, center)
            draw_trajectory(frame, tracker.trajectory)
            writer.writerow([frame_idx, center[0], center[1], visible])
        else:
            writer.writerow([frame_idx, -1, -1, 0])

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    csv_file.close()

    print(f"✅ Done: {video_name}")
    print(f"   → Video: {out_video_path}")
    print(f"   → CSV  : {csv_path}\n")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    video_files = [
        os.path.join(TEST_VIDEO_DIR, f)
        for f in os.listdir(TEST_VIDEO_DIR)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    ]

    if len(video_files) == 0:
        print("❌ No videos found in test_videos/")
    else:
        for video_path in sorted(video_files):
            process_video(video_path)

    print("🎯 All videos processed")
