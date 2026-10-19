import numpy as np

from .. import tools
from ..algo import Algo


class PPT(Algo):
    """Peak Price Tracking. Predicts that each asset's next price is its highest price
    over the last `window` periods and takes a fixed-length step from the last
    portfolio along the centered predicted price relatives, so weight moves towards
    assets that trade furthest below their recent peak.

    Reference:
        Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren and Ke-Kun Huang.
        A Peak Price Tracking-Based Learning System for Portfolio Selection, 2018.
        https://ieeexplore.ieee.org/document/7942104
        Author's code: https://github.com/laizhr/PPT
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=100, **kwargs):
        """
        :param window: Lookback window for the peak price, including the current price.
        :param eps: Step size, i.e. Euclidean distance between the last portfolio and
            the new one before it is projected onto the simplex.
        """

        super().__init__(min_history=1, **kwargs)

        # input check
        if window < 1:
            raise ValueError("window parameter must be >=1")
        if eps <= 0:
            raise ValueError("eps parameter must be >0")

        self.window = window
        self.eps = eps

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction; until a full window is available, use the
        # last price relative as the authors' code does
        if len(history) <= self.window:
            x_pred = history.iloc[-1] / history.iloc[-2]
        else:
            x_pred = self.predict(history.iloc[-self.window :])
        return self.update(last_b, x_pred)

    def predict(self, hist):
        """Predict next price relative as peak price over current price."""
        return hist.max() / hist.iloc[-1, :]

    def update(self, b, x_pred):
        """Step `eps` from `b` along the predicted price relatives projected onto the
        hyperplane sum(b) = 1, then project onto the simplex."""
        x_cent = x_pred - np.mean(x_pred)
        norm = np.linalg.norm(x_cent)

        # update portfolio
        if norm != 0:
            b = b + self.eps * x_cent / norm

        # project it onto simplex
        return tools.simplex_proj(b)


if __name__ == "__main__":
    tools.quickrun(PPT())
