import numpy as np


class Model():
    """
    The true model of the function.

    All methods and attributes that start with '_' should be treated as
    private.
    """

    def __init__(self, metamodel, kwargs):
        """ THIS IS A DUMMY MODEL. """
        self._dimension = 10
        return

    def set_random_seed(self, seed):
        """ This should set all used random number generator seeds. """
        np.random.seed(seed)
        return

    def get_dimension(self):
        """ Safe access for outside functions. """
        return self._dimension

    def evaluate(self, coords):
        """ THIS IS A DUMMY MODEL. """
        return coords[0] ** 2 + coords[1] ** 3 - 5 * coords[2]
