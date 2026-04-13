import pytest
import numpy as np

from services.state import AppState
from services.model_service import ModelService


@pytest.fixture
def state():
    return AppState()


@pytest.fixture
def fake_image():
    return np.random.rand(1, 64, 64, 3).astype("float32")


class FakeModel:
    def predict(self, x, verbose=0):
        batch = x.shape[0]
        out = np.zeros((batch, len(ModelService().model.model.output_shape[-1]) if hasattr(ModelService().model, "model") else 26))

        # always predict "A"
        out[:, 0] = 0.99
        return out


@pytest.fixture
def fake_model(monkeypatch):
    """
    replace real model on stable model
    """
    from services import model_service

    monkeypatch.setattr(model_service, "ModelService", lambda: FakeModel())
