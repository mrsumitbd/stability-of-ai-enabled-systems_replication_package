import numpy as np

#############################################################################
# Objective
# =========
# To tune the hyper-parameters of our model we need to define a model,
# decide which parameters to optimize, and define the objective function
# we want to minimize.

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

# gradient boosted trees tend to do well on problems like this
reg = GradientBoostingRegressor(n_estimators=50, random_state=0)

#############################################################################
# Next, we need to define the bounds of the dimensions of the search space
# we want to explore and pick the objective. In this case the cross-validation
# mean absolute error of a gradient boosting regressor over the Boston
# dataset, as a function of its hyper-parameters.

from skopt.space import Real, Integer
from skopt.utils import use_named_args


# The list of hyper-parameters we want to optimize. For each one we define the
# bounds, the corresponding scikit-learn parameter name, as well as how to
# sample values from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(1, 5, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set
# scikit-learn estimator parameters
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=None,
                                    scoring="neg_mean_absolute_error"))

#############################################################################
# Optimize all the things!
# ========================
# With these two pieces, we are now ready for sequential model-based
# optimisation. Here we use gaussian process-based optimisation.

from skopt import gp_minimize
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

#print("Best score=%.4f" % res_gp.fun)

print(f"\n\n\nScore: {res_gp.fun}\n\n\n")
