import numpy as np


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

    def __init__(self):
        """
        Initializes the Metamodel with the specified components, builds them
        and sets up the history of evaluations.
        """

        # Public attributes
        self.step_index = 0
        self.dimension = 10
        self.extra_data = 3
        self.history_size = 1000

        self.model = None
        self.relevator = None
        self.surrogate = None

        # Private attributes
        self._update_counter = 0
        self._update_interval = 500

        # Initiate components
        self._reset_history()

        return

    def _reset_history(self):
        """ Resets the history to an empty one. """
        self.history = np.zeros((self.history_size,
                                 self.dimension + self.extra_data))
        return

    def _update_history(self, params, relevance, is_relevant, prediction):
        """ Updates the history with a single evaluation. """
        return

    def _update(self, is_relevant):
        """ Update all components if needed. """
        self.step_index += 1
        self._update_counter += 1

        if self._update_counter > self._update_interval:
            self.surrogate.relearn(self.history)
            self.relevator.relearn(self.history)
            self._update_counter = 0

        return

    def evaluate(self, params):
        """ Evaluate the Metamodel at the given point. """
        # Determine relevance.
        relevance = self.relevator.evaluate(params)
        is_relevant = self.relevator.is_relevant(relevance)

        # Decide which prediction to use.
        if is_relevant:
            prediction = self.surrogate.evaluate(params)
        else:
            prediction = self.model.evaluate(params)

        # Update history and components.
        self._update_history(params, relevance, is_relevant, prediction)
        self._update(is_relevant)

        # Return result.
        return prediction
