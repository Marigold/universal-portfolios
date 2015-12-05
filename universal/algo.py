import sys
import numpy as np
import pandas as pd
import itertools
import logging
import inspect
import copy
from .result import AlgoResult, ListResult
from scipy.misc import comb
from . import tools


class Algo(object):
    """ Base class for algorithm calculating weights for online portfolio.
    You have to subclass either step method to calculate weights sequentially
    or weights method, which does it at once. weights method might be useful
    for better performance when using matrix calculation, but be careful about
    look-ahead bias.

    Upper case letters stand for matrix and lower case for vectors (such as
    B and b for weights).
    """

    # if true, replace missing values by last values
    REPLACE_MISSING = False

    # type of prices going into weights or step function
    #    ratio:  pt / pt-1
    #    log:    log(pt / pt-1)
    #    raw:    pt
    PRICE_TYPE = 'ratio'

    def __init__(self, min_history=None, frequency=1):
        """ Subclass to define algo specific parameters here.
        :param min_history: If not None, use initial weights for first min_window days. Use
            this if the algo needs some history for proper parameter estimation.
        :param frequency: algorithm should trade every `frequency` periods
        """
        self.min_history = min_history or 0
        self.frequency = frequency

    def init_weights(self, m):
        """ Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(m)

    def init_step(self, X):
        """ Called before step method. Use to initialize persistent variables.
        :param X: Entire stock returns history.
        """
        pass

    def step(self, x, last_b, history):
        """ Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        :param history: All returns up to now. You can omit this parameter to increase
            performance.
        """
        raise NotImplementedError('Subclass must implement this!')

    def _use_history_step(self):
        """ Use history parameter in step method? """
        step_args = inspect.getargspec(self.step)[0]
        return len(step_args) >= 4

    def weights(self, X, min_history=None, log_progress=True):
        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        min_history = self.min_history if min_history is None else min_history

        # init
        B = X.copy() * 0.
        last_b = self.init_weights(X.shape[1])
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)

        # use history in step method?
        use_history = self._use_history_step()

        # run algo
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.ix[t] = last_b

            # keep initial weights for min_history
            if t < min_history:
                continue

            # trade each `frequency` periods
            if (t + 1) % self.frequency != 0:
                continue

            # predict for t+1
            if use_history:
                history = X.iloc[:t+1]
                last_b = self.step(x, last_b, history)
            else:
                last_b = self.step(x, last_b)

            # convert last_b to suitable format if needed
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            # show progress by 10 pcts
            if log_progress:
                tools.log_progress(t, len(X), by=10)

        return B

    def _split_index(self, ix, nr_chunks, freq):
        """ Split index into chunks so that each chunk except of the last has length
        divisible by freq. """
        chunksize = int(len(ix) / freq / nr_chunks + 1) * freq
        return [ix[i*chunksize:(i+1)*chunksize] for i in range(len(ix) / chunksize + 1)]

    def run(self, S, n_jobs=1, log_progress=True):
        """ Run algorithm and get weights.
        :params S: Absolute stock prices. DataFrame with stocks in columns.
        :param show_progress: Log computation progress. Works only for algos with
            defined step method.
        :param n_jobs: run step method in parallel (step method can't depend on last weights)
        """
        if log_progress:
            logging.debug('Running {}...'.format(self.__class__.__name__))

        if isinstance(S, ListResult):
            P = S.to_dataframe()
        else:
            P = S

        # convert prices to proper format
        X = self._convert_prices(P, self.PRICE_TYPE, self.REPLACE_MISSING)

        # get weights
        if n_jobs == 1:
            try:
                B = self.weights(X, log_progress=log_progress)
            except TypeError:   # weights are missing log_progress parameter
                B = self.weights(X)
        else:
            with tools.mp_pool(n_jobs) as pool:
                ix_blocks = self._split_index(X.index, pool._processes * 2, self.frequency)
                min_histories = np.maximum(np.cumsum([0] + map(len, ix_blocks[:-1])) - 1, self.min_history)

                B_blocks = pool.map(_parallel_weights, [(self, X.ix[:ix_block[-1]], min_history, log_progress)
                                    for ix_block, min_history in zip(ix_blocks, min_histories)])

            # join weights to one dataframe
            B = pd.concat([B_blocks[i].ix[ix] for i, ix in enumerate(ix_blocks)])

        # cast to dataframe if weights return numpy array
        if not isinstance(B, pd.DataFrame):
            B = pd.DataFrame(B, index=P.index, columns=P.columns)

        if log_progress:
            logging.debug('{} finished successfully.'.format(self.__class__.__name__))

        # if we are aggregating strategies, combine weights from strategies
        # and use original assets
        if isinstance(S, ListResult):
            B = sum(result.B.mul(B[col], axis=0) for result, col in zip(S, B.columns))
            return AlgoResult(S[0].X, B)
        else:
            return AlgoResult(self._convert_prices(S, 'ratio'), B)

    def next_weights(self, S, last_b, **kwargs):
        """ Calculate weights for next day. """
        # use history in step method?
        use_history = self._use_history_step()
        history = self._convert_prices(S, self.PRICE_TYPE, self.REPLACE_MISSING)
        x = history.iloc[-1]

        if use_history:
            b = self.step(x, last_b, history, **kwargs)
        else:
            b = self.step(x, last_b, **kwargs)
        return pd.Series(b, index=S.columns)

    def run_subsets(self, S, r, generator=False):
        """ Run algorithm on all stock subsets of length r. Note that number of such tests can be
        very large.
        :param S: stock prices
        :param r: number of stocks in a subset
        :param generator: yield results
        """
        def subset_generator():
            total_subsets = comb(S.shape[1], r)

            for i, S_sub in enumerate(tools.combinations(S, r)):
                # run algorithm on given subset
                result = self.run(S_sub, log_progress=False)
                name = ', '.join(S_sub.columns.astype(str))

                # log progress by 1 pcts
                tools.log_progress(i, total_subsets, by=1)

                yield result, name
            raise StopIteration

        if generator:
            return subset_generator()
        else:
            results = []
            names = []
            for result, name in subset_generator():
                results.append(result)
                names.append(name)
            return ListResult(results, names)

    @classmethod
    def _convert_prices(self, S, method, replace_missing=False):
        """ Convert prices to format suitable for weight or step function.
        Available price types are:
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == 'raw':
            # normalize prices so that they start with 1.
            r = {}
            for name, s in S.iteritems():
                init_val = s.ix[s.first_valid_index()]
                r[name] = s / init_val
            X = pd.DataFrame(r)

            if replace_missing:
                X.ix[0] = 1.
                X = X.fillna(method='ffill')

            return X

        elif method == 'absolute':
            return S

        elif method in ('ratio', 'log'):
            # be careful about NaN values
            X = S / S.shift(1).fillna(method='ffill')
            X.ix[0] = 1.

            if replace_missing:
                X = X.fillna(1.)

            return np.log(X) if method == 'log' else X

        else:
            raise ValueError('invalid price conversion method')

    @classmethod
    def run_combination(cls, S, **kwargs):
        """ Get equity of algo using all combinations of parameters. All
        values in lists specified in kwargs will be optimized. Other types
        will be passed as they are to algo __init__ (like numbers, strings,
        tuples).
        Return ListResult object, which is basically a wrapper of list of AlgoResult objects.
        It is possible to pass ListResult to Algo or run_combination again
        to get AlgoResult. This is useful for chaining of Algos.

        Example:
            S = ...load data...
            list_results = Anticor.run_combination(S, alpha=[0.01, 0.1, 1.])
            result = CRP().run(list_results)

        :param S: Stock prices.
        :param kwargs: Additional arguments to algo.
        :param n_jobs: Use multiprocessing (-1 = use all cores). Use all cores by default.
        """
        if isinstance(S, ListResult):
            S = S.to_dataframe()

        n_jobs = kwargs.pop('n_jobs', -1)

        # extract simple parameters
        simple_params = {k: kwargs.pop(k) for k, v in kwargs.items()
                         if not isinstance(v, list)}

        # iterate over all combinations
        names = []
        params_to_try = []
        for seq in itertools.product(*kwargs.values()):
            params = dict(zip(kwargs.keys(), seq))

            # run algo
            all_params = dict(params.items() + simple_params.items())
            params_to_try.append(all_params)

            # create name with format param:value
            name = ','.join([str(k) + '=' + str(v) for k, v in params.items()])
            names.append(name)

        # try all combinations in parallel
        with tools.mp_pool(n_jobs) as pool:
            results = pool.map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])
        results = map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])

        return ListResult(results, names)

    def copy(self):
        return copy.deepcopy(self)


def _parallel_weights(tuple_args):
    self, X, min_history, log_progress = tuple_args
    try:
        return self.weights(X, min_history=min_history, log_progress=log_progress)
    except TypeError:   # weights are missing log_progress parameter
        return self.weights(X, min_history=min_history)


def _run_algo_params(tuple_args):
    S, cls, params = tuple_args
    logging.debug('Run combination of parameters: {}'.format(params))
    return cls(**params).run(S)
