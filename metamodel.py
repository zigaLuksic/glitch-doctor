import numpy as np
from model import Model
from history import History
from relevator import Relevator
from surrogate import Surrogate


class Metamodel():
    """
    The Metamodel has four significant elements, the model, the surrogate,
    the relevator and the history of evaluations.
    At every step, the metamodel uses the relevator to decide wheter to use the 
    model or the surrogate and afterwards updates the history and its 
    components.

    All methods and attributes that start with '_' should be treated as 
    private.
    """

    def __init__(self, model_kwargs, history_kwargs, relevator_kwargs, surrogate_kwargs):
        """ Initialize the Metamodel with the specified components. """

        # Public attributes
        self.step_index = 0
        self.model_evaluations = 0

        self.model = Model(self, model_kwargs)
        self.history = History(self, history_kwargs)
        self.relevator = Relevator(self, relevator_kwargs)
        self.surrogate = Surrogate(self, surrogate_kwargs)

        return

    def evaluate(self, coords):
        """ Evaluate the Metamodel at the given point. """

        # Determine relevance.
        relevance = self.relevator.evaluate(coords)
        is_relevant = self.relevator.is_relevant(relevance)

        # Decide which prediction to use.
        if is_relevant:
            prediction = self.model.evaluate(coords)
            self.model_evaluations += 1
        else:
            prediction = self.surrogate.evaluate(coords)

        # Update history and components.
        self.history.update(coords, prediction, relevance, is_relevant)

        # Update index and return result.
        self.step_index += 1
        return prediction
