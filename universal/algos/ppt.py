import numpy as np

from .. import tools
from ..algo import Algo


class PPT(Algo):
    """Peak Price Tracking. Predicts that each asset's next price is its highest price
    over the last `window` periods and takes a fixed-length step `eps` from the last
    portfolio along the centered predicted price relatives, then projects onto the
    simplex. Weight moves to the assets trading furthest below their recent peak.

    With the default eps = 100 the step dwarfs the simplex, so PPT holds a single
    asset on about 95% of days, the one furthest below its window peak, and switches
    it on more than half of the days. Like OLMAR and RPRT, without fees it beats
    uniform CRP on every bundled dataset we tried (4x on DJIA, 1e16x on NYSE), but
    the edge is gone at fees between 0.14% (SP500) and 0.6% (NYSE) per unit traded.

    The update is implemented from the authors' reference MATLAB code; the paper's
    text was not checked. The math, how it maps to this code and benchmarks are in
    ppt.md next to this file.

    Reference:
        Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren and Ke-Kun Huang.
        A Peak Price Tracking-Based Learning System for Portfolio Selection.
        IEEE Transactions on Neural Networks and Learning Systems 29(7), 2018.
        https://ieeexplore.ieee.org/document/7942104
        Authors' code: https://github.com/laizhr/PPT
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
        # predict next price relatives; for the first `window - 1` steps use the last
        # price relative, as the authors' code does
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
