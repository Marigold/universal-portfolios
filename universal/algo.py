import copy
import itertools
import logging

import numpy as np
import pandas as pd
from scipy.special import comb

from . import tools
from .result import AlgoResult, ListResult


class Algo:
    """Base class for algorithm calculating weights for online portfolio.
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
    PRICE_TYPE = "ratio"

    def __init__(self, min_history: int | None = None, frequency: int = 1, **kwargs):
        """Subclass to define algo specific parameters here.
        :param min_history: If not None, use initial weights for first min_window days. Use
            this if the algo needs some history for proper parameter estimation.
        :param frequency: algorithm should trade every `frequency` periods. `step` still
            sees every period: its `x` is the return accumulated since its previous call.
        """
        self.min_history = min_history or 0
        self.frequency = frequency

    def init_weights(self, columns):
        """Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(len(columns))

    def init_step(self, X):
        """Called before step method. Use to initialize persistent variables.
        :param X: Entire stock returns history.
        """

    def step(self, x, last_b, history=None):
        """Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        :param history: All returns up to now. You can omit this parameter to increase
            performance.
        """
        raise NotImplementedError("Subclass must implement this!")

    def weights(self, X, log_progress=True):
        """Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk."""
        B, _ = self._step_through(X, log_progress=log_progress)
        return B

    def _step_through(self, X, held_b=None, log_progress=True, **kwargs):
        """Call step over X. Return the weights held in each period of X and the weights
        for the period after X.
        :param held_b: Weights actually held in the last period of X, used for its step
            instead of the replayed ones.
        :param kwargs: Passed to the last step.
        """
        # init
        B = X.copy() * 0.0
        last_b = self.init_weights(X.columns)
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)

        # init step
        self.init_step(X)

        # return accumulated since the last step, for frequency > 1
        x_acc = None

        for t, (_, x) in enumerate(X.iterrows()):
            last = t == len(X) - 1
            if last and held_b is not None:
                last_b = held_b

            # save weights
            B.iloc[t] = last_b

            # keep initial weights for min_history
            if t < self.min_history:
                continue

            x_acc = x if x_acc is None else self._accumulate(x_acc, x)

            # trade each `frequency` periods
            if (t - self.min_history) % self.frequency != 0:
                continue

            # predict for t+1
            history = X.iloc[: t + 1]
            last_b = self.step(x_acc, last_b, history, **(kwargs if last else {}))
            x_acc = None

            # convert last_b to suitable format if needed
            if isinstance(last_b, np.matrix):
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            # show progress by 10 pcts
            if log_progress:
                tools.log_progress(t, len(X), by=10)

        return B, last_b

    def _accumulate(self, x_acc, x):
        """Combine the return accumulated so far with the next period's return."""
        if self.PRICE_TYPE == "ratio":
            return (x_acc * x).rename(x.name)
        if self.PRICE_TYPE == "log":
            return (x_acc + x).rename(x.name)
        # raw and absolute prices are levels, not returns
        return x

    def run(self, S, n_jobs=1, log_progress=True) -> AlgoResult:
        """Run algorithm and get weights.
        :params S: Absolute stock prices. DataFrame with stocks in columns.
        :param show_progress: Log computation progress. Works only for algos with
            defined step method.
        :param n_jobs: Removed, only 1 is accepted.
        """
        if n_jobs != 1:
            # it restarted the state and weights of every chunk, giving wrong weights
            # for most algorithms (see issue #149)
            raise ValueError("Running steps in parallel was removed, use n_jobs=1")

        if log_progress:
            logging.debug(f"Running {self.__class__.__name__}...")

        if isinstance(S, ListResult):
            P = S.to_dataframe()
        else:
            P = S

        # convert prices to proper format
        X = self._convert_prices(P, self.PRICE_TYPE, self.REPLACE_MISSING)

        # get weights
        try:
            B = self.weights(X, log_progress=log_progress)
        except TypeError:  # weights are missing log_progress parameter
            B = self.weights(X)

        # cast to dataframe if weights return numpy array
        if not isinstance(B, pd.DataFrame):
            B = pd.DataFrame(B, index=P.index, columns=P.columns)

        if log_progress:
            logging.debug(f"{self.__class__.__name__} finished successfully.")

        # if we are aggregating strategies, combine weights from strategies
        # and use original assets
        if isinstance(S, ListResult):
            B = sum(result.B.mul(B[col], axis=0) for result, col in zip(S, B.columns))
            return AlgoResult(S[0].X, B)
        else:
            return AlgoResult(self._convert_prices(S, "ratio"), B)

    def next_weights(self, S, last_b, **kwargs):
        """Calculate weights for next day. Replays the whole history through step, as
        algorithms that keep state between steps need it, so it takes as long as run.
        :param last_b: Weights held in the last period of S.
        :param kwargs: Passed to the last step.
        """
        X = self._convert_prices(S, self.PRICE_TYPE, self.REPLACE_MISSING)
        _, b = self._step_through(X, held_b=last_b, log_progress=False, **kwargs)
        return pd.Series(b, index=S.columns)

    def run_subsets(self, S, r, generator=False):
        """Run algorithm on all stock subsets of length r. Note that number of such tests can be
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
                name = ", ".join(S_sub.columns.astype(str))

                # log progress by 1 pcts
                tools.log_progress(i, total_subsets, by=1)

                yield result, name

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
    def _convert_prices(cls, S, method, replace_missing=False):
        """Convert prices to format suitable for weight or step function.
        Available price types are:
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == "raw":
            # normalize prices so that they start with 1.
            r = {}
            for name, s in S.items():
                init_val = s.loc[s.first_valid_index()]
                r[name] = s / init_val
            X = pd.DataFrame(r)

            if replace_missing:
                X.iloc[0] = 1.0
                X = X.ffill()

            return X

        elif method == "absolute":
            return S

        elif method in ("ratio", "log"):
            # be careful about NaN values
            X = S / S.shift(1).ffill()
            for name, s in X.items():
                ix = s.first_valid_index()
                assert ix is not None, f"No valid index for {name}"
                X[name].iloc[s.index.get_loc(ix) - 1] = 1.0

            if replace_missing:
                X = X.fillna(1.0)

            return np.log(X) if method == "log" else X

        else:
            raise ValueError("invalid price conversion method")

    @classmethod
    def run_combination(cls, S, **kwargs):
        """Get equity of algo using all combinations of parameters. All
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

        n_jobs = kwargs.pop("n_jobs", -1)

        # extract simple parameters
        simple_params = {
            k: kwargs.pop(k)
            for k in tuple(kwargs.keys())
            if not isinstance(kwargs[k], list)
        }

        # iterate over all combinations
        names = []
        params_to_try = []
        for seq in itertools.product(*kwargs.values()):
            params = dict(zip(kwargs.keys(), seq))

            # run algo
            all_params = dict(list(params.items()) + list(simple_params.items()))
            params_to_try.append(all_params)

            # create name with format param:value
            name = ",".join([str(k) + "=" + str(v) for k, v in params.items()])
            names.append(name)

        # try all combinations in parallel
        with tools.mp_pool(n_jobs) as pool:
            results = pool.map(
                _run_algo_params, [(S, cls, all_params) for all_params in params_to_try]
            )

        return ListResult(results, names)

    def copy(self):
        return copy.deepcopy(self)


def _run_algo_params(tuple_args):
    S, cls, params = tuple_args
    logging.debug(f"Run combination of parameters: {params}")
    return cls(**params).run(S)
