import numpy as np


class Surrogate():
    """
    Learns and evaluates a surrogate model based on evaluations of the true
    model.

    All methods and attributes that start with '_' should be treated as 
    private.
    """

    def __init__(self):

        self._predictor = None
        return

    def _update(self):
        return
    
    def _prepare_data(self, params):
        return params

    def evaluate(self, params):
        data = self._prepare_data(params)
        prediction = self._predictor.predict(data)
        return prediction
