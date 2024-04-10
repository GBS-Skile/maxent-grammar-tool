from typing import List

import numpy as np
from numpy.typing import NDArray

from model.priors import Priors


class GaussianRegularizer:
    mu: NDArray[np.float64]  # (p,)
    sig_sq: NDArray[np.float64]  # (p,)

    def __init__(self, priors: Priors, keys: List[str]):
        gaussians = [priors.get(k) for k in keys]
        self.mu = np.array([g.mean for g in gaussians])
        self.sig_sq = np.array([g.var for g in gaussians])

    def score(self, beta):
        """returns -logP"""
        return (0.5 * np.square(beta - self.mu) / self.sig_sq).sum()

    def grad(self, beta):
        """returns d/db (-logP)"""
        return (beta - self.mu) / self.sig_sq
