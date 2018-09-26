import numpy as np
from sklearn import ensemble

class Relevator():
    """
    Predicts the relevance of points using machine learning methods.

    All methods and attributes that start with '_' should be treated as 
    private.
    """

    def __init__(self, metamodel, kwargs):
        self.metamodel = metamodel
        return

    def _update(self):
        return

    def evaluate(self, coords):
        return 0

    def is_relevant(self, relevance):
        return np.random.rand(1) < 0.5
