from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

from services.camera_service import CameraService
from services.state import AppState
from services.model_service import ModelService
from utils.image_processing import detect_skin, enhance_hand_roi
from utils.face import FaceMasker

app = Flask(__name__)

state = AppState()
camera = CameraService()
model = ModelService()
face_masker = FaceMasker()


def resize_keep_aspect(img, width=640):
    h, w = img.shape[:2]
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h)), scale


def draw_debug_panel(frame, gesture, conf, stable_count, required, target):
    lines = [
        f"Gesture: {gesture}",
        f"Confidence: {conf:.2f}",
        f"Target: {target}",
        f"Match: {gesture == target}",
        f"Stable: {stable_count}/{required}",
        f"Verdict: {state.last_verdict}",
    ]

    y = 140
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, y + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


def generate_frames():
    REQUIRED_CONF = 0.75
    STABLE = 8

    stable_gesture = None
    stable_count = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        display, scale = resize_keep_aspect(frame, 640)
        proc = frame.copy()

        if state.mask_faces:
            proc = face_masker.mask_faces(proc)

        proc = cv2.resize(proc, (640, 360))

        skin, contours, _ = detect_skin(proc)
        hand = skin > 0.03

        gesture = "nothing"
        conf = 0.0

        # ===== DETECTION =====
        if hand and contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)

            if w > 50 and h > 50:
                roi = proc[y:y+h, x:x+w]
                roi = enhance_hand_roi(roi)

                img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                gesture, conf = model.predict(img)

                dx = int(x * scale)
                dy = int(y * scale)
                dw = int(w * scale)
                dh = int(h * scale)

                cv2.rectangle(display, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 2)

        if hand and conf > REQUIRED_CONF and gesture != "nothing":
            if stable_gesture == gesture:
                stable_count += 1
            else:
                stable_gesture = gesture
                stable_count = 1
        else:
            stable_count = max(0, stable_count - 1)

        if state.last_verdict is None:
            if stable_count >= STABLE:
                state.last_verdict = (
                    "correct" if gesture == state.current_target else "incorrect"
                )

                print(
                    f"[RESULT] gesture={gesture} "
                    f"target={state.current_target} "
                    f"conf={conf:.2f}"
                )

        cv2.putText(display, f"Target: {state.current_target}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        status = "Show gesture"

        if state.last_verdict == "correct":
            status = "CORRECT"
        elif state.last_verdict == "incorrect":
            status = f"Expected {state.current_target}"

        cv2.putText(display, status, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        draw_debug_panel(
            display,
            gesture,
            conf,
            stable_count,
            STABLE,
            state.current_target
        )

        ret, buffer = cv2.imencode(".jpg", display)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html", target=state.current_target)


@app.route("/learning")
def learning():
    return render_template("learning.html", letters=state.class_names)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/next_letter")
def next_letter():
    state.next_letter()
    return jsonify({"target": state.current_target})


@app.route("/retry_letter")
def retry():
    state.reset()
    state.last_verdict = None
    return jsonify({"status": "reset"})


@app.route("/toggle_mask")
def toggle_mask():
    state.mask_faces = not state.mask_faces
    return jsonify({"status": state.mask_faces})


if __name__ == "__main__":
    print("[START] Flask server launching...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
