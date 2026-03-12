import pytest
import numpy as np
import os
from utils import detect_skin, FaceMasker, enhance_hand_roi


@pytest.fixture
def empty_frame():
    return np.zeros((360, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_frame():
    return np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)


@pytest.fixture
def face_masker():
    return FaceMasker()


@pytest.mark.utils
def test_face_masker_creation(face_masker):
    assert face_masker is not None


@pytest.mark.utils
def test_enhance_hand_roi(empty_frame):
    result = enhance_hand_roi(empty_frame)
    assert result is not None
    assert result.shape == empty_frame.shape


@pytest.mark.utils
def test_detect_skin_basic(empty_frame):
    skin_percentage, contours, mask = detect_skin(empty_frame)
    assert isinstance(skin_percentage, (int, float))
    assert 0 <= skin_percentage <= 1
    assert mask.shape[:2] == (360, 640)


@pytest.mark.utils
@pytest.mark.skipif(
    not os.path.exists("asl_model.h5"),
    reason="model file not found"
)
def test_detect_skin_with_model_context(random_frame):
    skin_percentage, contours, mask = detect_skin(random_frame)
    assert isinstance(skin_percentage, (int, float))
    assert 0 <= skin_percentage <= 1