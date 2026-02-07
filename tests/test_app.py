import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, class_names, current_target

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """checking the main page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<html" in response.data
    assert b"Gesture training" in response.data

def test_video_feed_route(client):
    """checking the generator of video"""
    response = client.get('/video_feed')
    assert response.status_code == 200
    assert response.mimetype.startswith("multipart/")

def test_next_letter(client):
    """checking the next letter after click on the button"""
    global current_target
    initial = current_target
    response = client.get('/next_letter')
    assert response.status_code == 200
    data = response.get_json()
    assert data['target'] in class_names
    assert data['target'] != initial

def test_retry_letter(client):
    """resetting the last click"""
    response = client.get('/retry_letter')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'reset'

def test_toggle_mask(client):
    """switching face mask"""
    response = client.get('/toggle_mask')
    assert response.status_code == 200
    assert response.get_json()['status'] in ['on', 'off']

def test_gesture_image(client):
    """checking of the getting guest image"""
    response = client.get('/gesture_image/H')
    assert response.status_code == 200
    data = response.get_json()
    assert 'gestures/H.jpg' in data['url']

def test_learning_page(client):
    """learning page"""
    response = client.get('/learning')
    assert response.status_code == 200
    for letter in class_names:
        assert letter.encode() in response.data