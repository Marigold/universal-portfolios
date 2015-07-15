from ..algo import Algo
from universal.algos import CRP
from .. import tools
import numpy as np


class BestSoFar(Algo):
    """ Algorithm selects asset that had the best performance in last x days.
    """
    PRICE_TYPE = 'ratio'

    def __init__(self, n=None, metric='return', min_history=None, **metric_kwargs):
        self.n = n
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        super(BestSoFar, self).__init__(min_history=min_history)

    def init_weights(self, m):
        # use uniform weights until you get enough history
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # get history
        hist = history.iloc[-self.n:] if self.n else history

        # choose best performing asset
        if self.metric == 'return':
            p = hist.prod()
        elif self.metric == 'sharpe':
            p = hist.apply(lambda s: tools.sharpe(np.log(s), **self.metric_kwargs))

        # select only one asset randomly
        p += 1E-10 * np.random.randn(len(p))

        return (p == p.max()).astype(float)


if __name__ == '__main__':
    tools.quickrun(BestSoFar(metric='sharpe'))
