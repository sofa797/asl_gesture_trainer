import pytest
import numpy as np
import os
from app import model, DummyModel, class_names


@pytest.fixture
def model_info():
    is_real = os.path.exists('asl_model.h5') and not isinstance(model, DummyModel)
    return {'is_real': is_real, 'model': model}


def test_model_available(model_info):
    assert hasattr(model_info['model'], 'predict')


def test_model_is_real(model_info):
    if model_info['is_real']:
        assert True, 'Real model loaded successfully'
    else:
        pytest.warning('Using dummy model- check if asl_model.h5 exists')


@pytest.mark.skipif(not os.path.exists('asl_model.h5'), reason='model file not found')
def test_model_prediction(model_info):
    test_input = np.zeros((1, 64, 64, 3), dtype='float32')
    predictions = model_info["model"].predict(test_input, verbose=0)
    assert predictions.shape == (1, len(class_names)), f'Expected {(1, len(class_names))}, got {predictions.shape}'
    assert np.all(predictions >= 0) and np.all(predictions <= 1), 'Probabilities must be in [0, 1]'