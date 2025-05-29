import types

import numpy as np
import pandas as pd
from scipy import stats

from universal import tools


class AssetFilter(object):
    def __init__(self, window=None, threshold=0.3):
        self.window = window
        self.threshold = threshold

    def _filter(self, R):
        # sh[col] = tools.sharpe(total_ret.div(total_weights, axis=0), alpha=0.000001)
        # to_remove = set(sh.index[sh - full_sharpe > 0.00001])

        SAMPLES = 50
        np.random.seed(42)
        sh = []
        for _ in range(SAMPLES):
            # get bootstrap sample
            R_sample = R.sample(n=len(R), replace=True)
            sh.append(
                {col: tools.sharpe(R_sample[col], alpha=0.00001) for col in R_sample}
            )
        sh = pd.DataFrame(sh)

        sh_diff = sh.subtract(sh["full"], 0)
        cdf = stats.norm.cdf(
            0.0, loc=sh_diff.mean(), scale=0.01 + sh_diff.std() / np.sqrt(len(sh_diff))
        )
        to_remove = sh_diff.columns[cdf < self.threshold]
        to_remove = to_remove.drop(columns=["full"], errors="ignore")

        print(list(to_remove))
        return to_remove

    def fit(self, R, B):
        # convert it to log returns
        R_log = np.log(R)

        if self.window:
            R_log = R_log.iloc[-self.window :]

        # find sharpe ratio without assets
        RR = {"full": R_log.sum(1)}
        for col in R.columns:
            total_ret = R_log.drop(columns=[col]).sum(1)
            # total_weights = B.drop(columns=[col]).sum(1) + 1e-10
            RR[col] = total_ret

        to_remove = self._filter(pd.DataFrame(RR))

        # print(to_remove)
        return to_remove


def filter_result(S, algo, asset_filter=None, result=None):
    """Filter assets for algo by their past-performance."""
    result = result or algo.run(S)
    asset_filter = asset_filter or AssetFilter()

    # monkey-patch algo's step
    step_fun = algo.step

    def step(self, x, last_b, history):
        # find assets to remove -asset_r is already weighted
        R = result.asset_r.loc[: x.name]
        B = result.B.loc[: x.name]
        cols = asset_filter.fit(R, B)

        # get weights with removed assets
        w = step_fun(
            x.drop(columns=cols),
            last_b.drop(columns=cols),
            history.drop(columns=cols, axis=1),
        )

        # put back assets with zero weights
        w = w.reindex(last_b.index).fillna(0.0)
        return w

    algo.step = types.MethodType(step, algo)

    # run algo with filtered assets
    new_result = algo.run(S)

    # put back old step method
    algo.step = types.MethodType(step_fun, algo)
    return new_result, result
