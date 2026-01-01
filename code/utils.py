import cv2

# def draw_trajectory(frame, trajectory):
#     for i in range(1, len(trajectory)):
#         cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)

# def draw_centroid(frame, center):
#     cx, cy = center
#     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


import cv2

def draw_trajectory(frame, trajectory):
    for i in range(1, len(trajectory)):
        pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
        pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

def draw_centroid(frame, center):
    if center is None:
        return

    cx, cy = int(center[0]), int(center[1])
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
