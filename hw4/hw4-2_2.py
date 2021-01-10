#!/usr/bin/python
import numpy as np
import mnist_reader as mr
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from ConfigSpace.conditions import InCondition
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO


X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')


X_train = X_train[:2000]
y_train = y_train[:2000]



def bagging_from_cfg(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    decision_tree = DecisionTreeClassifier(max_depth=int(cfg['max_depth']))
    clf = BaggingClassifier(base_estimator=decision_tree, n_estimators=int(cfg['n_estimators']), random_state=42)

    scores = cross_val_score(clf, X_train, y_train, cv=5)

    return 1-np.mean(scores)  # Minimize!


cs = ConfigurationSpace()
# C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
n_estimators = UniformFloatHyperparameter("n_estimators", 10, 100, default_value=10.0)
max_depth = UniformFloatHyperparameter("max_depth", 1, 100, default_value=1.0)

cs.add_hyperparameter(n_estimators)
cs.add_hyperparameter(max_depth)

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 50,   # max. number of function evaluations; for this example set to a low number
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=bagging_from_cfg)

incumbent = smac.optimize()
inc_value = bagging_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))