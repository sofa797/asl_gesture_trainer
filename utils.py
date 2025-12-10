import cv2
import numpy as np
import mediapipe as mp

def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 5)
    skin_percentage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return skin_percentage, contours, mask

class FaceMasker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.4
        )
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.use_haar = True
        
    def mask_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_haar:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
                    cv2.rectangle(frame, (max(0, x-10), max(0, y-10)), 
                                (min(frame.shape[1], x+w+10), min(frame.shape[0], y+h+10)), 
                                (0, 0, 0), -1)
                return frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 0), -1)
        return frame

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
        print(f"Ошибка при улучшении изображения: {e}")
        return hand_roi