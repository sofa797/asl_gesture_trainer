import cv2
import numpy as np
from .timing import timing


@timing("detect_skin")
def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 5)

    skin_percentage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[-2]

    if contours is None:
        contours = []
    if not isinstance(contours, list):
        contours = list(contours)

    return skin_percentage, contours, mask


@timing("enhance_hand_roi")
def enhance_hand_roi(hand_roi):
    if hand_roi.size == 0:
        return hand_roi

    try:
        ycrcb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])

        ycrcb = cv2.merge(channels)
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        return enhanced

    except Exception as e:
        print(f"enhance error: {e}")
        return hand_roi
