import numpy as np

from .. import tools
from ..algo import Algo


class OLMAR(Algo):
    """On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10, alpha=0.5, ma_type="SMA"):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        :param alpha: Decaying factor for using EMA as price relative prediction
        :param ma_type: Type of moving average used, either SMA or EMA
        """

        super().__init__(min_history=1)

        # input check
        if window < 2:
            raise ValueError("window parameter must be >=3")
        if eps < 1:
            raise ValueError("epsilon parameter must be >=1")
        if ma_type not in ["SMA", "EMA"]:
            raise ValueError('ma_type should be either "SMA" or "EMA"')

        self.window = window
        self.eps = eps
        self.alpha = alpha
        self.ma_type = ma_type

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        # self.x_pred = X.iloc[1, :] / X.iloc[0, :]
        self.x_pred = X.iloc[0, :]

    def step(self, x, last_b, history):
        # calculate return prediction
        if len(history) < self.window + 1:
            x_pred = history.iloc[-1]
        else:
            x_pred = self.predict(x, history.iloc[-self.window :])
        b = self.update(last_b, x_pred, self.eps)
        return b

    def predict(self, x, hist):
        """Predict next price relative."""
        if self.ma_type == "SMA":
            return hist.mean() / hist.iloc[-1, :]
        else:
            real_x = hist.iloc[-1, :] / hist.iloc[-2, :]
            x_pred = self.alpha + (1 - self.alpha) * np.divide(self.x_pred, real_x)
            self.x_pred = x_pred
            return x_pred

    def update(self, b, x_pred, eps):
        """Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights."""
        x_pred_mean = np.mean(x_pred)
        excess_return = x_pred - x_pred_mean
        denominator = (excess_return * excess_return).sum()
        if denominator != 0:
            lam = max(0.0, (eps - np.dot(b, x_pred)) / denominator)
        else:
            lam = 0

        # update portfolio
        b = b + lam * (excess_return)

        # project it onto simplex
        return tools.simplex_proj(b)


if __name__ == "__main__":
    tools.quickrun(OLMAR())
