from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd


@dataclass
class Gaussian:
    mean: np.float64
    var: np.float64
    default: ClassVar["Gaussian"]


Gaussian.default = Gaussian(0.0, 100_000.0)


class Priors:
    def __init__(
        self, *, filename: Optional[str] = None, default: Gaussian = Gaussian.default
    ):
        if filename:
            df = pd.read_csv(
                filename,
                delimiter="\t",
                header=None,
                names=["abbrev", "mu", "sig2"],
                dtype=dict(mu=np.float64, sig2=np.float64),
            )
            self.features = dict(
                df.apply(
                    lambda row: (row.abbrev, Gaussian(row.mu, row.sig2)), axis=1
                ).tolist()
            )
        else:
            self.features = {}
        self.default = default

    def get(self, key: str) -> Gaussian:
        return self.features.get(key, self.default)
