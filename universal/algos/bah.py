import numpy as np

from .. import tools
from ..algo import Algo


class BAH(Algo):
    """Buy and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever."""

    PRICE_TYPE = "raw"

    def __init__(self, b=None):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """
        super().__init__()
        self.b = b

    def weights(self, S):
        """Weights function optimized for performance."""
        if self.b is None:
            b = np.array([0 if s == "CASH" else 1 for s in S.columns])
            b = b / b.sum()
        else:
            b = self.b

        # weights are proportional to price times initial weights
        w = S.shift(1) * b

        # normalize
        w = w.div(w.sum(axis=1), axis=0)

        w.iloc[0] = 1.0 / S.shape[1]

        return w


if __name__ == "__main__":
    tools.quickrun(BAH())
