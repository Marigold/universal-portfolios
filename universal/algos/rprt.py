# -*- coding: utf-8 -*-
import numpy as np

from .. import tools
from ..algo import Algo


class RPRT(Algo):
    """Reweighted Price Relative Tracking System for Automatic Portfolio Optimization
    Reference:
        Zhao-Rong Lai and Pei-Yi Yang and Liangda Fang and Xiaotian Wu.
        Reweighted Price Relative Tracking System for Automatic Portfolio Optimization, 2018.
        https://ieeexplore.ieee.org/document/8411138
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=50, theta=0.8, **kwargs):
        """
        :param window: Lookback window.
        :param eps:
        :param theta:
        """

        super().__init__(min_history=1, **kwargs)

        # input check
        if window < 2:
            raise ValueError("window parameter must be >=3")

        self.window = window
        self.eps = eps
        self.theta = theta
        self.phi = np.array([])

    def init_step(self, X):
        # set initial phi to x1
        self.phi = X.iloc[1, :] / X.iloc[0, :]

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(history.iloc[-self.window :])

        # compute the reweighted price relative
        D_pred = np.diag(np.array(x_pred))
        last_phi = self.phi
        last_price_relative = history.iloc[-1, :] / history.iloc[-2, :]
        gamma_pred = (
            self.theta
            * last_price_relative
            / (self.theta * last_price_relative + last_phi)
        )
        phi_pred = gamma_pred + np.multiply(
            1 - gamma_pred, np.divide(last_phi, last_price_relative)
        )
        self.phi = phi_pred

        # Update the weights
        b = self.update(b=last_b, phi_pred=phi_pred, D_pred=D_pred)

        return b

    def predict(self, hist):
        """Predict next price relative."""
        return hist.mean() / hist.iloc[-1, :]

    def update(self, b, phi_pred, D_pred):
        # Calculate variables
        phi_pred_mean = np.mean(phi_pred)

        if np.linalg.norm(phi_pred - phi_pred_mean) ** 2 == 0:
            lam = 0
        else:
            lam = max(
                0.0,
                (self.eps - np.dot(b, phi_pred))
                / np.linalg.norm(phi_pred - phi_pred_mean) ** 2,
            )

        # update portfolio
        if lam != 0:  # avoid numerical problem (0 * inf)
            b_ = b + lam * np.dot(D_pred, (phi_pred - phi_pred_mean))
        else:
            b_ = b

        b_ = np.clip(b_, -1e10, 1e10)  # avoid numerical problem

        # project it onto simplex
        return tools.simplex_proj(y=b_)


if __name__ == "__main__":
    tools.quickrun(RPRT(window=5, eps=50, theta=0.8))
