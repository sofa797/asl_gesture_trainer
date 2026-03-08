import pytest
import numpy as np
import os
from app import model, DummyModel, class_names


@pytest.fixture
def model_info():
    is_real = os.path.exists('asl_model.h5') and not isinstance(model, DummyModel)
    return {
        "is_real": is_real,
        "model": model
    }


@pytest.mark.model
def test_model_available(model_info):
    """model has predict method"""
    assert hasattr(model_info["model"], "predict")


@pytest.mark.model
def test_model_is_real(model_info):
    """check if real model is loaded"""
    if model_info["is_real"]:
        assert True
    else:
        pytest.skip("Using dummy model - asl_model.h5 not found")


@pytest.mark.model
@pytest.mark.skipif(
    not os.path.exists("asl_model.h5"),
    reason="model file not found"
)
def test_model_prediction(model_info):
    """test prediction output shape"""
    test_input = np.zeros((1, 64, 64, 3), dtype="float32")
    predictions = model_info["model"].predict(test_input, verbose=0)
    assert predictions.shape == (1, len(class_names))
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)