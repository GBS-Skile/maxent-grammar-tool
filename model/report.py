from dataclasses import dataclass
from typing import List, Tuple

from numpy import linalg as LA
from scipy.optimize import OptimizeResult

from model.tableaux import Tableaux

@dataclass
class Candidate:
    input: str
    output: str
    probability: float
    frequency: float


@dataclass
class Report:
    tableaux: Tableaux
    opt_result: OptimizeResult

    def weights(self) -> List[Tuple[str, float]]:
        return list(zip(self.tableaux.feature_names, self.opt_result.x))

    def candidates(self) -> List[Candidate]:
        result = []
        for ex in self.tableaux.examples:
            probs = ex.predict(self.opt_result.x)
            for i, output in enumerate(ex.surfaces):
                result.append(
                    Candidate(
                        ex.underlying,
                        output,
                        probs[i],
                        ex.freqs[i] / ex.freqs.sum(),
                    )
                )
        return result

    def grad_norm(self):
        return LA.norm(self.opt_result.jac)
