from ultralytics import YOLO

class BallDetector:
    def __init__(self, model_path, conf=0.05):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        if results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
            return None

        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy
