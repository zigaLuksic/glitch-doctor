import numpy as np


class History():
    """
    Keeps the history of evaluations, but only of the TRUE MODEL and
    all the RELEVANCE predictions.

    Every line is of the form:
    [coords_0, ..., coords_(dim - 1), value, relevance, is_relevant, step_index]

    It keeps a fix amount of space and overwrites old entries when full.
    Managed by the '_write_index' so only use safe access to get correctly
    ordered entries.

    All methods and attributes that start with '_' should be treated as 
    private.
    """

    def __init__(self, metamodel, kwargs):
        self.metamodel = metamodel

        self.size = kwargs.get("size", 1000)

        self._dimension = metamodel.model.get_dimension() + 1
        self._data = np.zeros((self.size, self._dimension))
        self._rel_data = np.zeros(self.size)
        self._write_index = 0   # Used for more efficient updating.
        self._rel_write_index = 0
        return

    def update(self, coords, prediction, relevance, is_relevant):
        # Update the history of relevance.
        self._rel_data[self._rel_write_index] = relevance
        self._rel_write_index = (self._rel_write_index + 1) % self.size
        # Update the history of model evaluations (if needed).
        if is_relevant:
            line = self._write_index
            self._data[line, 0:len(coords)] = np.array(coords)
            self._data[line, -1] = prediction
            self._write_index = (self._write_index + 1) % self.size
        return

    def _order_data(self, data, write_index):
        roll = self.size - write_index
        ordered = np.roll(data, roll, axis=0)
        return ordered

    def get_model_evaluations(self):
        """ 
        Returns only the coords and values for evaluations of the actual model.
        """
        return self._order_data(self._data, self._write_index)

    def get_relevance_predictions(self, relevance):
        """
        Returns only the relevance predictions.
        """
        return self._order_data(self._rel_data, self._rel_write_index)

    def get_whole_history(self):
        return self.get_model_evaluations()

# class All_History():
#    """
#    Keeps the history of ALL evaluations. Cannot guarantee a certain amount of
#    model evaluations.
#
#    Every line is of the form:
#    [coords_0, ..., coords_(dim - 1), value, relevance, is_relevant]
#
#    It keeps a fix amount of space and overwrites old entries when full.
#    Managed by the '_write_index' so only use safe access to get correctly
#    ordered entries.
#
#    All methods and attributes that start with '_' should be treated as
#    private.
#    """
#
#    def __init__(self, metamodel, kwargs):
#        self.metamodel = metamodel
#
#        self.size = kwargs.get("size", 1000)
#
#        self._dimension = metamodel.model.get_dimension() + 3
#        self._index_of = {
#            "prediction": self._dimension - 3,
#            "relevance": self._dimension - 2,
#            "is_relevant": self._dimension - 1
#        }
#        self._data = np.zeros((self.size, self._dimension))
#        self._write_index = 0   # Used for more efficient updating.
#        return
#
#    def update(self, coords, prediction, relevance, is_relevant):
#        line = self._write_index
#
#        self._data[line, 0:len(coords)] = np.array(coords)
#        self._data[line, self._index_of["prediction"]] = prediction
#        self._data[line, self._index_of["relevance"]] = relevance
#        self._data[line, self._index_of["is_relevant"]] = float(is_relevant)
#
#        # Readjust the write index.
#        self._write_index = (self._write_index + 1) % self.size
#        return
#
#    def _get_ordered_data(self):
#        roll = self.size - self._write_index
#        ordered_data = np.roll(self._data, roll, axis=0)
#        return ordered_data
#
#    def get_model_evaluations(self):
#        """
#        Returns only the coords and values for evaluations of the actual model.
#        """
#        data = self._get_ordered_data()
#        model_eval_indices = data[:, self._index_of["is_relevant"]] < 0.5
#        model_history = data[model_eval_indices, :]
#        return model_history
#
#    def get_relevance_predictions(self, relevance):
#        """
#        Returns only the relevance predictions.
#        """
#        data = self._get_ordered_data()
#        relevance_history = data[:, self._index_of["relevance"]]
#        return relevance_history
#
#    def get_whole_history(self):
#        return np.copy(self._data)
#
