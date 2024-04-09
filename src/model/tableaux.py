from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Feature:
    full_name: str
    abbreviation: str

    @staticmethod
    def from_col(col: pd.Series):
        return Feature(col[0], col[1])


@dataclass
class Example:
    underlying: str
    surfaces: List[str]  # (n,)
    freqs: NDArray[np.float64]  # (n,)
    violations: NDArray[np.float64]  # (n, p)

    def predict(self, beta):
        a = np.exp(self.violations @ beta)
        return a / a.sum()

    def score(self, beta):
        """returns -log(P)"""
        z = self.predict(beta)
        return (-np.log(z) * self.freqs).sum()

    def grad(self, beta):
        z = self.predict(beta)
        return (
            -(self.freqs @ self.violations) + (z @ self.violations) * self.freqs.sum()
        )


@dataclass
class Tableaux:
    features: List[Feature]
    examples: List[Example]

    @staticmethod
    def load(filename: str) -> "Tableaux":
        """
        Format: Bruce Hayes's OTSoft format
        https://linguistics.ucla.edu/people/hayes/otsoft/
        """
        df = pd.read_csv(filename, delimiter="\t", header=None)
        features = df.iloc[0:2, 3:].apply(Feature.from_col).tolist()

        has_input = ~pd.isna(df.iloc[:, 0])
        indice = df[has_input].index.tolist()

        underlyings = df[has_input][0].tolist()
        indice.append(len(df))
        examples = []
        for i in range(len(indice) - 1):
            tabular = df.iloc[indice[i] : indice[i + 1], 1:].fillna(0)
            examples.append(
                Example(
                    underlyings[i],
                    tabular.iloc[:, 0].tolist(),
                    tabular.iloc[:, 1].to_numpy(),
                    tabular.iloc[:, 2:].to_numpy().astype(np.float64),
                )
            )

        return Tableaux(features, examples)

    @property
    def feature_names(self) -> List[str]:
        return [f.abbreviation for f in self.features]

    def score(self, beta):
        return sum(ex.score(beta) for ex in self.examples)

    def grad(self, beta):
        return sum(ex.grad(beta) for ex in self.examples)
