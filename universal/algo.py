# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import itertools
import logging
import inspect
from result import AlgoResult, ListResult


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


    def __init__(self, min_history=None):
        """ Subclass to define algo specific parameters here.
        :param min_history: If not None, use initial weights for first min_window days. Use
            this if the algo needs some history for proper parameter estimation.
        """
        self.min_history = min_history or 0
        

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


    def weights(self, X):
        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        # init
        B = X.copy() * 0.
        last_b = self.init_weights(X.shape[1])
        
        # use history parameter in step method?
        step_args = inspect.getargspec(self.step)[0]
        use_history = len(step_args) >= 4

        # run algo
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.ix[t] = last_b
            
            # keep initial weights for min_history
            if t < self.min_history:
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
            progress = 10 * int(10. * t / len(X))
            if not hasattr(self, '_progress') or progress != self._progress:
                self._progress = progress
                logging.debug('Progress: {}%...'.format(progress))

        return B


    def run(self, S):
        """ Run algorithm and get weights.
        :params S: Absolute stock prices. DataFrame with stocks in columns.
        :param show_progress: Show computation progress. Works only for algos with
            defined step method.
        """
        logging.debug('Running {}...'.format(self.__class__.__name__))

        if isinstance(S, ListResult):
            P = S.to_dataframe()
        else:
            P = S

        # get weights
        X = self._convert_prices(P, self.PRICE_TYPE, self.REPLACE_MISSING)
        B = self.weights(X)
        
        # cast to dataframe if weights return numpy array
        if not isinstance(B, pd.DataFrame):
            B = pd.DataFrame(B, index=P.index, columns=P.columns)
            
        logging.debug('{} finished successfully.'.format(self.__class__.__name__))

        # if we are aggregating strategies, combine weights from strategies
        # and use original assets
        if isinstance(S, ListResult):
            B = sum(result.B.mul(B[col], axis=0) for result, col in zip(S, B.columns))
            return AlgoResult(S[0].X, B)
        else:
            return AlgoResult(self._convert_prices(S, 'ratio'), B)
    
    
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
        """
        if isinstance(S, ListResult):
            S = S.to_dataframe()

        # extract simple parameters
        simple_params = {k: kwargs.pop(k) for k,v in kwargs.items()
                                          if not isinstance(v, list)}

        results = []        # list of AlgoResult
        names = []
        for seq in itertools.product(*kwargs.values()):
            params = dict(zip(kwargs.keys(), seq))

            # run algo
            all_params = dict(params.items() + simple_params.items())
            logging.debug('Run combination of parameters: {}'.format(params))
            result = cls(**all_params).run(S)
            results.append(result)

            # create name in format param:value
            name = ','.join([str(k) + '=' + str(v) for k, v in params.items()])
            names.append(name)

        return ListResult(results, names)

