import cv2
import time

class CameraService:
    def __init__(self):
        self.cap = self.open_camera()

    def open_camera(self):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        for backend in backends:
            for i in range(5):
                cap = cv2.VideoCapture(i, backend)

                if not cap.isOpened():
                    cap.release()
                    continue

                time.sleep(0.3)
                for _ in range(10):
                    cap.read()

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)

                #print(f"Camera opened: index={i}, backend={backend}")
                return cap

                cap.release()

        print("No camera found")
        return None

    def read(self):
        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()

        if not ok or frame is None:
            print("frame ok:", ok)
            return False, None

        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
