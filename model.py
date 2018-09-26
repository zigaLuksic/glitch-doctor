import numpy as np


class Model():
    """
    The true model of the function.

    All methods and attributes that start with '_' should be treated as 
    private.
    """

    def __init__(self, metamodel, kwargs):
        self._dimension = 3  # DUMMY
        return

    def get_dimension(self):
        """ Safe access for outside functions. """
        return self._dimension

    def evaluate(self, coords):
        return coords[0] ** 2
