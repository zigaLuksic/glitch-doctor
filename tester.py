import numpy as np
from metamodel import Metamodel

metamodel_kwargs = {"random_seed": 28537}
model_kwargs = {}
surrogate_kwargs = {"rebuild_interval": 100}
threshold_kwargs = {"desired_surr_rate": 0.7,
                    "acceptable_offset": 0.05,
                    "step": 0.001,
                    "alpha": 42,
                    "beta": 10}
relevator_kwargs = {"rebuild_interval": 100,
                    "threshold_kwargs": threshold_kwargs}
history_kwargs = {"size": 1000,
                  "use_size": 100}

mm = Metamodel(metamodel_kwargs, model_kwargs, surrogate_kwargs,
               relevator_kwargs, history_kwargs)


for _ in range(10000):
    coords = 10 * np.random.rand(10)
    mm.evaluate(coords)

print(mm.history._use_data)
print(mm.history.get_model_usage_rate())
print(mm.relevator._threshold.value)
