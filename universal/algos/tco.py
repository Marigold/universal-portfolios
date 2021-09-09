import numpy as np
import numpy.typing as npt

from .. import tools
from ..algo import Algo


class TCO(Algo):
    """Transaction costs optimization. The TCO algorithm needs just a next return prediction
    to work, see the paper for more details.

    Paper : https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?referer=&httpsredir=1&article=4761&context=sis_research
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, trx_fee_pct=0, eta=10, **kwargs):
        """
        :param trx_fee_pct: transaction fee in percent
        :param eta: smoothing parameter
        """
        super().__init__(**kwargs)
        self.trx_fee_pct = trx_fee_pct
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, p, last_b, history):
        # calculate return prediction
        x_pred = self.predict(p, history)
        x = p / history.iloc[-2]
        b = self.update_tco(x, last_b, x_pred)
        return b

    def predict(self, p, history) -> npt.NDArray:
        """Predict returns on next day.
        :param p: raw price
        """
        raise NotImplementedError()

    def update_tco(self, x: npt.NDArray, b: npt.NDArray, x_pred: npt.NDArray):
        """
        :param x: ratio of change in price
        """
        lambd = 10 * self.trx_fee_pct

        # last price adjusted weights
        updated_b = np.multiply(b, x) / np.dot(b, x)

        # Calculate variables
        vt = x_pred / np.dot(updated_b, x_pred)
        v_t_ = np.mean(vt)

        # Update portfolio
        b_1 = self.eta * (vt - np.dot(v_t_, 1))
        b_ = updated_b + np.sign(b_1) * np.maximum(
            np.zeros(len(b_1)), np.abs(b_1) - lambd
        )

        # project it onto simplex
        proj = tools.simplex_proj(y=b_)
        return proj


class TCO1(TCO):
    def __init__(self, type="reversal", **kwargs):
        self.type = type
        super().__init__(min_history=1, **kwargs)

    def predict(self, p, history):
        if self.type == "reversal":
            return history.iloc[-2] / p
        elif self.type == "trend":
            return p / history.iloc[-2]
        else:
            raise NotImplementedError()


class TCO2(TCO):
    def __init__(self, window=5, **kwargs):
        # input check
        if window < 2:
            raise ValueError("window parameter must be >=3")

        super().__init__(min_history=window, **kwargs)

        self.window = window

    def predict(self, p, history):
        # OLMAR style prediction
        return (history.iloc[-self.window :] / p).mean()


if __name__ == "__main__":
    tools.quickrun(TCO1())
