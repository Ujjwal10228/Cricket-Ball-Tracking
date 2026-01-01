import numpy as np
from kalman import KalmanFilterCV

class BallTracker:
    def __init__(self, fps, frame_height):
        self.kf = KalmanFilterCV(1 / fps)
        self.H = frame_height

        self.initialized = False
        self.consec_valid = 0
        self.prev_center = None
        self.frames_since_detection = 0
        self.trajectory = []

        self.MIN_MOTION_PX = 5
        self.INIT_CONSEC_FRAMES = 3
        self.MIN_DETECTIONS_FOR_TRACK = 5
        self.MAX_PRED_FRAMES = 1

        self.detection_count = 0

    def step(self, detection):
        visible = 0
        center = None

        # ---------- Initialization ----------
        if not self.initialized:
            if detection:
                self.consec_valid += 1
                self.prev_center = detection
                self.detection_count += 1

                if (self.consec_valid >= self.INIT_CONSEC_FRAMES and
                        self.detection_count >= self.MIN_DETECTIONS_FOR_TRACK):
                    self.kf.x[:2] = np.array(detection).reshape(2, 1)
                    self.kf.x[2:] = 0
                    self.initialized = True
                    self.trajectory.clear()
            return None, 0

        # ---------- Tracking ----------
        if detection:
            motion = np.linalg.norm(np.array(detection) - np.array(self.prev_center))
            if motion >= self.MIN_MOTION_PX:
                self.kf.update(detection)
                center = detection
                visible = 1
                self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection <= self.MAX_PRED_FRAMES:
                # center = self.kf.predict()
                pred = self.kf.predict()
                center = (int(pred[0]), int(pred[1]))

                visible = 0
            else:
                self.reset()
                return None, 0

        if center is not None:
            self.prev_center = center
            self.trajectory.append(center)

        return center, visible

    def reset(self):
        self.initialized = False
        self.consec_valid = 0
        self.prev_center = None
        self.frames_since_detection = 0
        self.detection_count = 0
        self.trajectory.clear()
