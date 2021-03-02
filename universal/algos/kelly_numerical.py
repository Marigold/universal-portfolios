from typing import Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats
from cvxopt import solvers
from scipy import optimize

from .. import tools
from ..algo import Algo

solvers.options['show_progress'] = False


def _g(f, vals_norm, rfr_tta, N):
    return np.sum(np.log(f * (vals_norm - rfr_tta) + rfr_tta + 1.)) / N


def _g_der(f, vals_norm, rfr_tta, N):
    # print(vals_norm.shape)
    return np.sum((vals_norm - rfr_tta) / (f * (vals_norm - rfr_tta) + rfr_tta + 1.)) / N


def _test_ks_2samp(vals1: np.array, vals2: np.array):
    vals2n = vals2[-len(vals1):]
    res = scipy.stats.ks_2samp(vals1, vals2n)
    return res.pvalue > 0.05


class KellyNumerical(Algo):
    vals_norm_prev = None

    def __init__(self, coeff: float = 0.5, window_large: int = 100, window_small: int = 20,
                 limits: Optional[Tuple[float, float]] = None,
                 cooldown: int = 2):
        super().__init__()
        self.coeff = coeff
        self.window_large = window_large
        self.window_small = window_small
        self.limits = limits
        if limits is not None:
            assert limits[0] < limits[1]
        self.current_cooldown = None
        self.cooldown = cooldown
        self.broker_fee_coef = 1. / cooldown

    def init_step(self, x):
        pass

    def step(self, x, last_b, history=None):
        if len(history) < self.window_large:
            return 0.0

        if self.current_cooldown is not None:
            self.current_cooldown -= 1
            if self.current_cooldown <= 0:
                self.current_cooldown = None
            return last_b

        kelly_coef = self.coeff

        broker_fee = 0.1 / 100.0
        trading_ticks = 252
        risk_free_rate = 0.005  # 0.0179 # 1 year treasury rate
        rfr_tta = ((1. + risk_free_rate) ** (1. / trading_ticks) - 1.)  # risk_free_rate__trading_ticks_adjusted

        vals = history.iloc[-self.window_large:, 0].values
        w_size = self.window_small
        f_star_semi_prev = last_b or 0.0
        semi_kelly_param = self.coeff

        vals_log_inc = np.diff(np.log(vals))
        N = len(vals_log_inc)

        vals_inc = np.diff(vals) / vals[:-1]
        df = pd.DataFrame(dict(vals_inc=vals_inc))
        means = df.vals_inc.rolling(w_size).mean().dropna().to_numpy()[:-1]
        stds = df.vals_inc.rolling(w_size).std().dropna().to_numpy()[:-1]
        vals_norm = ((vals_inc[w_size:] - means) / stds) * stds[-1] + means[-1]

        f_prev = last_b or 0.0

        if -broker_fee * self.broker_fee_coef <= _g_der(f_prev, vals_norm, rfr_tta, N) <= broker_fee * self.broker_fee_coef:
            # f_res = optimize.newton(lambda x: g_der(x) - cn, 0.001, maxiter=1000)
            # res = f_prev * kelly_coef
            return last_b

        else:
            if self.vals_norm_prev is not None:
                # if test_anderson_ksamp(vals_norm[:rebalance_time_prev], vals_norm):
                if _test_ks_2samp(self.vals_norm_prev, vals_norm):
                    # res = f_prev * kelly_coef
                    # res = last_b[0]
                    return last_b

            f_res = optimize.minimize(
                method='COBYLA',
                fun=lambda f: -_g(f, vals_norm, rfr_tta, N) + broker_fee * self.broker_fee_coef * np.abs(f - f_prev),
                x0=1.0,
                # x0=f_prev,
                constraints=[
                    dict(
                        type='ineq',
                        fun=lambda f: f * (vals_norm.max() * 1.15 - rfr_tta) + rfr_tta + 1.,
                    ),
                    dict(
                        type='ineq',
                        fun=lambda f: f * (vals_norm.min() * 1.15 - rfr_tta) + rfr_tta + 1.,
                    ),
                ],
            )
            res = f_res.x * kelly_coef
            self.vals_norm_prev = vals_norm

        if self.limits is not None:
            res = max(self.limits[0], min(self.limits[1], res))

        return [res]


if __name__ == '__main__':
    tools.quickrun(algo=KellyNumerical())
