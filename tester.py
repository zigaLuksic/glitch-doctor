import numpy as np
from metamodel import Metamodel

history_kwargs = {"size": 100}
relevator_kwargs = {}
surrogate_kwargs = {"rebuild_interval": 100}
model_kwargs = {}

mm = Metamodel(model_kwargs, history_kwargs,
               relevator_kwargs, surrogate_kwargs)


for _ in range(1000):
    coords = 10 * np.random.rand(3)
    mm.evaluate(coords)

print(mm.history.get_whole_history())
