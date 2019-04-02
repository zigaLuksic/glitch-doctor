import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, differential_evolution
from metamodel import Metamodel
from sklearn import ensemble, tree

# -----------------------------------------------------------------------------
# These are all the default values of the MetaModel
# -----------------------------------------------------------------------------
metamodel_kwargs = {"random_seed": 28537}

model_kwargs = {"dimension": 1,
                "function": lambda x: 0}

surrogate_kwargs = {"rebuild_interval": 100,
                    "predictor": ensemble.RandomForestRegressor()}

threshold_kwargs = {"type": "alpha-beta",
                    "desired_surr_rate": 0.7,
                    "acceptable_offset": 0.05,
                    "step": 0.0001,
                    "alpha": 42,
                    "beta": 10}

relevator_kwargs = {"rebuild_interval": 100,
                    "threshold_kwargs": threshold_kwargs,
                    "predictor": ensemble.RandomForestRegressor()}

history_kwargs = {"size": 500,
                  "use_size": 200}
# -----------------------------------------------------------------------------

# Example setup for optimisation

# Set up the optimisation algorithm
seed = 28537
testfun = rosen
d = 15
bounds = [(-5, 5) for i in range(d)]

# Set up the meta model
metamodel_kwargs = {"seed": seed}
model_kwargs = {"dimension": d,
                "function": testfun}
metamodel = Metamodel(metamodel_kwargs, model_kwargs, surrogate_kwargs,
                      relevator_kwargs, history_kwargs)

# Wrap function so that it keeps history
pure_history = []
def history_fun(x):
    result = testfun(x)
    pure_history.append((result, 1))
    return result

np.random.seed(seed)
result = differential_evolution(history_fun, bounds, maxiter=50, tol=0.000001)
print(result.x, result.fun)
print(len(pure_history))

# Also wrap metamodel so that we get history
mm_history = []
def history_mm(x):
    result = metamodel.evaluate(x)
    # Find out wheter the model was used
    i = metamodel.history._use_write_index - 1
    model_used = metamodel.history._use_data[i]
    mm_history.append((result, model_used))
    return result

np.random.seed(seed)
result = differential_evolution(history_mm, bounds, maxiter=150, tol=0.000001)

print(result.x, result.fun)
print(metamodel.history.get_model_usage_rate())
print(metamodel.model_evaluations)


# Plot metamodel procedure
mm_history = mm_history[:]
surr_y = [v for (v, r) in mm_history if r == 0]
mod_y = [v for (v, r) in mm_history if r == 1]
surr_x = [i for (i, (v, r)) in enumerate(mm_history) if r == 0]
mod_x = [i for (i, (v, r)) in enumerate(mm_history) if r == 1]

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(mod_x, mod_y, s=2, c='b', marker=".", label='model')
ax1.scatter(surr_x, surr_y, s=2, c='r', marker=".", label='surrogate')
plt.legend(loc='upper left')
plt.show()
