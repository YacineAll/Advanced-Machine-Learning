import optuna
import torch.nn as nn


def objective(trial):
    #  TODO: 

study = optuna.create_study()
study.optimize(objective, n_trials=20)
print(study.best_params)
