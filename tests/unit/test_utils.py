import numpy as np
from utils.image_processing import detect_skin, enhance_hand_roi


def test_detect_skin_runs():
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    skin, contours, mask = detect_skin(img)

    assert mask.shape == (100, 100)
    assert isinstance(contours, list)


def test_enhance_empty_roi():
    empty = np.array([], dtype=np.uint8)
    result = enhance_hand_roi(empty)

    assert result is not None
