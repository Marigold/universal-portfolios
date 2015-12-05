from ..algo import Algo
from .. import tools
import numpy as np


class DynamicCRP(Algo):
    # use logarithm of prices
    PRICE_TYPE = 'ratio'

    def __init__(self, n=None, min_history=None, **kwargs):
        self.n = n
        self.opt_weights_kwargs = kwargs
        if min_history is None:
            if n is None:
                min_history = 252
            else:
                min_history = n
        super(DynamicCRP, self).__init__(min_history=min_history)

    def init_weights(self, m):
        self._importances = np.zeros(m)

        # use uniform weights until you get enough history
        return self.opt_weights_kwargs.get('max_leverage', 1.) * np.ones(m) / m

    def step(self, x, last_b, history):
        # update frequency
        self.opt_weights_kwargs['freq'] = tools.freq(history.index)

        hist = history.iloc[-self.n:] if self.n else history

        ws = tools.opt_weights(hist, **self.opt_weights_kwargs)
        return ws
