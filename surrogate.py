import numpy as np


class Surrogate():
    """
    Learns and evaluates a surrogate model based on evaluations of the true
    model.

    All methods and attributes that start with '_' should be treated as 
    private.
    """

    def __init__(self, metamodel, kwargs):

        self._predictor = None
        return

    def _update(self):
        return
    
    def _prepare_data(self, coords):
        return coords

    def evaluate(self, coords):
        data = self._prepare_data(coords)
        prediction = self._predictor.predict(data)
        return prediction
