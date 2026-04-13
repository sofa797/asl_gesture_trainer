import os
import numpy as np
import keras

CLASS_NAMES = [
 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
 'R','S','T','U','V','W','X','Y','Z'
]

class Dummy:
    def predict(self, x):
        batch = x.shape[0]
        out = np.random.rand(batch, len(CLASS_NAMES)) * 0.3
        for i in range(batch):
            out[i, np.random.randint(0, len(CLASS_NAMES))] = 0.9
        return out


class ModelService:
    def __init__(self):
        self.model = self._load()

    def _load(self):
        if os.path.exists("asl_model.h5"):
            return keras.models.load_model("asl_model.h5", compile=False)
        return Dummy()

    def predict(self, img):
        pred = self.model.predict(img)[0]
        idx = int(np.argmax(pred))

        if idx >= len(CLASS_NAMES):
            idx = len(CLASS_NAMES) - 1

        return CLASS_NAMES[idx], float(pred[idx])
