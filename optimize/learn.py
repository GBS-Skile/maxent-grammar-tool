import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize

from model.tableaux import Tableaux
from optimize.regularizer import GaussianRegularizer


def objective(beta, tbx: Tableaux, reg: GaussianRegularizer):
    return tbx.score(beta) + reg.score(beta)


def gradient(beta, tbx: Tableaux, reg: GaussianRegularizer):
    return tbx.grad(beta) + reg.grad(beta)


def learn(beta0, tbx, reg) -> OptimizeResult:
    return minimize(
        objective,
        beta0,
        args=(tbx, reg),
        jac=gradient,
        bounds=Bounds(-np.inf, 0.0),
        tol=1e-10,
    )
