import pytest
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os



MODEL_PATH = "asl_model.h5"
TEST_DIR = "static/gestures"
CLASS_NAMES =   ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', 'del', 'nothing', 'space']


@pytest.mark.model
@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH) or not os.listdir(TEST_DIR),
    reason="model or test images are not found"
)
def test_model_on_real_images():
    # arrange
    model = load_model(MODEL_PATH, compile=False)
    X_test = []
    for file in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
        X_test.append(img)

    X_test = np.array(X_test)
    if X_test.size == 0:
        pytest.skip("images do not exist")

    # act
    predictions = model.predict(X_test)

    # assert
    assert predictions.shape[1] == len(CLASS_NAMES)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)