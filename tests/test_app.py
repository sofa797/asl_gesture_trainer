import pytest
import app as app_module
from app import app, class_names, current_target

@pytest.mark.api
def test_index_route(client):
    """checking the main page"""
    response = client.get('/')
    assert b"<html" in response.data
    assert b"Gesture training" in response.data


@pytest.mark.api
def test_video_feed_route(client):
    """checkinmg the generator of video"""
    response = client.get('/video_feed')
    assert response.status_code == 200
    assert response.mimetype.startswith("multipart/")


@pytest.mark.api
def test_next_letter(client):
    """checking the next letter after click on the button"""
    initial = current_target
    response = client.get('/next_letter')
    assert response.status_code == 200
    data = response.get_json()
    assert data['target'] in class_names
    assert data['target'] != initial


@pytest.mark.api
def test_retry_letter(client):
    """resetting the last click"""
    response = client.get('/retry_letter')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'reset'


@pytest.mark.api
def test_toggle_mask(client):
    """switching face mask"""
    response = client.get('/toggle_mask')
    assert response.status_code == 200
    assert response.get_json()['status'] in ['on', 'off']


@pytest.mark.api
def test_gesture_image(client):
    """checking of the getting gesture image"""
    response = client.get('/gesture_image/H')
    assert response.status_code == 200
    data = response.get_json()
    assert 'gestures/H.jpg' in data['url']


@pytest.mark.api
def test_learning_page(client):
    """learning page"""
    response = client.get('/learning')
    assert response.status_code == 200
    for letter in class_names:
        assert letter.encode() in response.data


@pytest.mark.api
def test_generate_frames_with_contour(monkeypatch):
    """gesture detection"""

    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    contour = np.array([[[10,10]],[[100,10]],[[100,100]],[[10,100]]])
    class FakeCap:
        def isOpened(self):
            return True
        def read(self):
            return True, frame
    monkeypatch.setattr(app_module, "cap", FakeCap())
    monkeypatch.setattr(
        app_module,
        "detect_skin",
        lambda img: (0.1, [contour], None)
    )
    monkeypatch.setattr(
        app_module.face_masker,
        "mask_faces",
        lambda img: img
    )
    monkeypatch.setattr(
        app_module,
        "enhance_hand_roi",
        lambda img: img
    )
    gen = app_module.generate_frames()
    frame = next(gen)

    assert b'--frame' in frame