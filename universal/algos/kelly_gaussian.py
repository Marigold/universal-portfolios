from typing import Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats
from cvxopt import solvers

from .. import tools
from ..algo import Algo

solvers.options['show_progress'] = False


def _bartletts_test_equality_variances(vals1: np.array, vals2: np.array):
    n = len(vals1) + len(vals2)
    k = 2

    group_n = np.array([len(vals1), len(vals2)])
    group_variance = np.array([np.var(vals1), np.var(vals2)])

    pool_var = 1 / (n - k) * np.sum((group_n - 1) * group_variance)

    x2_num = (n - k) * np.log(pool_var) - np.sum((group_n - 1) * np.log(group_variance))
    x2_den = 1 + 1 / (3 * (k - 1)) * (np.sum(1 / (group_n - 1)) - 1 / (n - k))

    x2 = x2_num / x2_den

    p = 1 - scipy.stats.chi2.cdf(x2, k - 1)

    return p > 0.05


def _ttest_equality_means(vals1: np.array, vals2: np.array):
    res = scipy.stats.ttest_ind(vals1, vals2)
    return res.pvalue > 0.05


class KellyGaussian(Algo):
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
            return [0.0]

        if self.current_cooldown is not None:
            self.current_cooldown -= 1
            if self.current_cooldown <= 0:
                self.current_cooldown = None
            return last_b

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
        df = pd.DataFrame(dict(vals_log_inc=vals_log_inc))
        means = df.vals_log_inc.rolling(w_size).mean().dropna().to_numpy()[:-1]
        stds = df.vals_log_inc.rolling(w_size).std().dropna().to_numpy()[:-1]
        vals_norm = (vals_log_inc[w_size:] - means) / stds

        vals_norm_mean = np.mean(vals_norm)
        vals_norm_std = np.std(vals_norm)

        vals_mean = vals_norm_mean * stds[-1] + means[-1]
        vals_std = vals_norm_std * stds[-1]

        mu_log = vals_mean
        sigma_log = vals_std

        mu = mu_log + sigma_log ** 2 / 2
        sigma = sigma_log

        f_star_prev = f_star_semi_prev[0] * semi_kelly_param
        f_star_buy = (mu - rfr_tta - broker_fee * self.broker_fee_coef) / sigma ** 2
        f_star_sell = (mu - rfr_tta + broker_fee * self.broker_fee_coef) / sigma ** 2

        if (f_star_prev < f_star_buy or f_star_prev > f_star_sell) \
                and self.vals_norm_prev is not None \
                and _bartletts_test_equality_variances(self.vals_norm_prev, vals_norm) \
                and _ttest_equality_means(self.vals_norm_prev, vals_norm):
            return last_b

        if f_star_prev < f_star_buy:
            f_star = f_star_buy
            self.vals_norm_prev = vals_norm
            self.current_cooldown = self.cooldown
        elif f_star_prev > f_star_sell:
            f_star = f_star_sell
            self.vals_norm_prev = vals_norm
            self.current_cooldown = self.cooldown
        else:
            f_star = f_star_prev

        f_star_semi = f_star / semi_kelly_param

        if self.limits is not None:
            f_star_semi = max(self.limits[0], min(self.limits[1], f_star_semi))

        return [f_star_semi]


if __name__ == '__main__':
    tools.quickrun(algo=KellyGaussian())
