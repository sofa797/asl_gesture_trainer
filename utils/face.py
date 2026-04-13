import cv2
import mediapipe as mp
from .timing import timing


class FaceMasker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.4
        )

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.use_haar = True

    @timing("mask_faces")
    def mask_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_haar:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
                cv2.rectangle(frame,
                              (max(0, x-10), max(0, y-10)),
                              (min(frame.shape[1], x+w+10),
                               min(frame.shape[0], y+h+10)),
                              (0, 0, 0), -1)
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if results.detections:
            h, w, _ = frame.shape

            for d in results.detections:
                bbox = d.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(w - x, width + 2 * padding)
                height = min(h - y, height + 2 * padding)

                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 0), -1)

        return frame
