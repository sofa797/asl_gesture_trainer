import pytest
import numpy as np
from services.model_service import ModelService


@pytest.mark.unit
def test_model_predict_shape():
    model = ModelService()
    x = np.random.rand(1, 64, 64, 3).astype("float32")

    gesture, conf = model.predict(x)

    assert isinstance(gesture, str)
    assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize("idx", list(range(5)))
@pytest.mark.unit
def test_random_inputs(idx):
    model = ModelService()
    x = np.random.rand(1, 64, 64, 3).astype("float32")

    gesture, conf = model.predict(x)

    assert gesture is not None
