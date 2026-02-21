from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import numpy as np
import os
import keras

app = Flask(__name__)
MODEL_PATH = 'asl_model.h5'
class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', 'del', 'nothing', 'space']

class DummyModel:
    def predict(self, X, verbose=None):
        batch_size = X.shape[0]
        predictions = np.random.rand(batch_size, len(class_names)) * 0.3
        for i in range(batch_size):
            idx = np.random.randint(0, len(class_names))
            predictions[i, idx] = np.random.uniform(0.7, 0.95)
        return predictions

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"Model not found: {MODEL_PATH}")
        model = DummyModel()
except Exception as e:
    print(f"Load error: {e}")
    model = DummyModel()

if not isinstance(model, DummyModel):
    try:
        test = np.zeros((1, 64, 64, 3), dtype='float32')
        pred = model.predict(test, verbose=0)
        print(f"Real model OK | Output: {pred.shape} | Max: {np.max(pred):.3f}")
    except Exception as e:
        print(f"Test failed: {e}")
        model = DummyModel()

from utils import detect_skin, FaceMasker, enhance_hand_roi
face_masker = FaceMasker()

cap = None
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
except:
    pass

current_target = 'A'
last_verdict = None
mask_faces = True

def generate_frames():
    global current_target, last_verdict, mask_faces
    REQUIRED_CONFIDENCE = 0.75
    STABLE_FRAMES_REQUIRED = 8
    stable_gesture = None
    stable_count = 0
    
    if cap is None or not cap.isOpened():
        while True:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Target: {current_target}", (200, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, "Camera not available", (150, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        original_frame = cv2.flip(frame, 1)
        display_frame = cv2.resize(original_frame, (640, 360))
        processing_frame = original_frame.copy()
        
        if mask_faces:
            processing_frame = face_masker.mask_faces(processing_frame)
        processing_frame = cv2.resize(processing_frame, (640, 360))

        skin_percentage, contours, _ = detect_skin(processing_frame)
        hand_detected = skin_percentage > 0.03
        gesture = "nothing"
        confidence = 0.0

        if hand_detected and contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w > 50 and h > 50:
                padding = 15
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(processing_frame.shape[1] - x, w + 2 * padding)
                h = min(processing_frame.shape[0] - y, h + 2 * padding)
                hand_roi = processing_frame[y:y+h, x:x+w]

                if hand_roi.size > 0:
                    enhanced_roi = enhance_hand_roi(hand_roi)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    img_rgb = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (64, 64))
                    img_normalized = img_resized.astype('float32') / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)

                    predictions = model.predict(img_input, verbose=0)[0]
                    predicted_idx = np.argmax(predictions)
                    confidence = float(predictions[predicted_idx])
                    gesture = class_names[predicted_idx] if predicted_idx < len(class_names) else 'nothing'

        if hand_detected and gesture != 'nothing' and confidence > REQUIRED_CONFIDENCE:
            if stable_gesture == gesture:
                stable_count += 1
            else:
                stable_gesture = gesture
                stable_count = 1
        else:
            stable_gesture = None
            stable_count = 0

        if stable_count >= STABLE_FRAMES_REQUIRED:
            last_verdict = 'correct' if gesture == current_target else 'incorrect'
        elif not hand_detected:
            last_verdict = None

        cv2.putText(display_frame, f"Target: {current_target}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if last_verdict == 'correct':
            status, color = "CORRECT", (0, 255, 0)
        elif last_verdict == 'incorrect':
            status, color = f"Expected: {current_target}", (0, 0, 255)
        elif hand_detected and stable_gesture:
            status, color = f"Letter: {stable_gesture} ({stable_count}/{STABLE_FRAMES_REQUIRED})", (0, 255, 255)
        else:
            status, color = "Show a gesture", (200, 200, 200)

        cv2.putText(display_frame, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', target=current_target)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next_letter')
def next_letter():
    global current_target, last_verdict
    idx = class_names.index(current_target)
    current_target = class_names[(idx + 1) % len(class_names)]
    last_verdict = None
    return {"target": current_target}

@app.route('/retry_letter')
def retry_letter():
    global last_verdict
    last_verdict = None
    return {"status": "reset"}

@app.route('/toggle_mask')
def toggle_mask():
    global mask_faces
    mask_faces = not mask_faces
    return {"status": "on" if mask_faces else "off"}

@app.route('/gesture_image/<letter>')
def gesture_image(letter):
    if letter.upper() in class_names:
        return {"url": url_for('static', filename=f'gestures/{letter.upper()}.jpg')}
    return {"url": url_for('static', filename='gestures/placeholder.jpg')}

@app.route('/learning')
def learning_page():
    return render_template('learning.html', letters=class_names)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cap is not None:
            cap.release()