import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from universal import tools


class PickleMixin(object):

    def save(self, filename):
        """ Save object as a pickle """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        """ Load pickled object. """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class AlgoResult(PickleMixin):
    """ Results returned by algo's run method. The class containts useful
    metrics such as sharpe ratio, mean return, drawdowns, ... and also
    many visualizations.
    You can specify transactions by setting AlgoResult.fee. Fee is
    expressed in a percentages as a one-round fee.
    """

    def __init__(self, X, B):
        """
        :param X: Price relatives.
        :param B: Weights.
        """
        # set initial values
        self._fee = 0.
        self._B = B
        self.rf_rate = 0.
        self._X = X

        # update logarithms, fees, etc.
        self._recalculate()

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        self._X = _X
        self._recalculate()

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B
        self._recalculate()

    @property
    def fee(self):
        return self._fee

    @fee.setter
    def fee(self, value):
        """ Set transaction costs. Fees can be either float or Series
        of floats for individual assets with proper indices. """
        if isinstance(value, dict):
            value = pd.Series(value)
        if isinstance(value, pd.Series):
            missing = set(self.X.columns) - set(value.index)
            assert len(missing) == 0, 'Missing fees for {}'.format(missing)

        self._fee = value
        self._recalculate()

    def _recalculate(self):
        # calculate return for individual stocks
        r = (self.X - 1) * self.B
        self.asset_r = r + 1
        self.r = r.sum(axis=1) + 1

        # stock went bankrupt
        self.r[self.r < 0] = 0.

        # add fees
        if not isinstance(self._fee, float) or self._fee != 0:
            fees = (self.B.shift(-1).mul(self.r, axis=0) - self.B * self.X).abs()
            fees.iloc[0] = self.B.ix[0]
            fees.iloc[-1] = 0.
            fees *= self._fee

            self.asset_r -= fees
            self.r -= fees.sum(axis=1)

        self.r_log = np.log(self.r)

    @property
    def weights(self):
        return self.B

    @property
    def equity(self):
        return self.r.cumprod()

    @property
    def equity_decomposed(self):
        """ Return equity decomposed to individual assets. """
        return self.asset_r.cumprod()

    @property
    def asset_equity(self):
        return self.X.cumprod()

    @property
    def total_wealth(self):
        return self.r.prod()

    @property
    def profit_factor(self):
        x = self.r_log
        up = x[x > 0].sum()
        down = -x[x < 0].sum()
        return up / down if down != 0 else np.inf

    @property
    def sharpe(self):
        """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year.
        """
        return tools.sharpe(self.r_log, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def information(self):
        """ Information ratio benchmarked against uniform CRP portfolio. """
        s = self.X.mean(axis=1)
        x = self.r_log - np.log(s)

        mu, sd = x.mean(), x.std()

        freq = self.freq()
        if sd > 1e-8:
            return mu / sd * np.sqrt(freq)
        elif mu > 1e-8:
            return np.inf * np.sign(mu)
        else:
            return 0.

    @property
    def growth_rate(self):
        return self.r_log.mean() * self.freq()

    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r_log.std()

    @property
    def annualized_return(self):
        return np.exp(self.r_log.mean() * self.freq()) - 1

    @property
    def annualized_volatility(self):
        return np.exp(self.r_log).std() * np.sqrt(self.freq())

    @property
    def drawdown_period(self):
        ''' Returns longest drawdown perid. Stagnation is a drawdown too. '''
        x = self.equity
        period = [0.] * len(x)
        peak = 0
        for i in range(len(x)):
            # new peak
            if x[i] > peak:
                peak = x[i]
                period[i] = 0
            else:
                period[i] = period[i-1] + 1
        return max(period) * 252. / self.freq()

    @property
    def max_drawdown(self):
        ''' Returns highest drawdown in percentage. '''
        x = self.equity
        return max(1. - x / x.cummax())

    @property
    def winning_pct(self):
        x = self.r_log
        win = (x > 0).sum()
        all_trades = (x != 0).sum()
        return float(win) / all_trades

    def freq(self, x=None):
        """ Number of data items per year. If data does not contain
        datetime index, assume daily frequency with 252 trading days a year."""
        x = x or self.r
        return tools.freq(x.index)

    def summary(self, name=None):
        return """Summary{}:
    Profit factor: {:.2f}
    Sharpe ratio: {:.2f}
    Information ratio (wrt UCRP): {:.2f}
    Annualized return: {:.2f}%
    Annualized volatility: {:.2f}%
    Longest drawdown: {:.0f} days
    Max drawdown: {:.2f}%
    Winning days: {:.1f}%
        """.format(
            '' if name is None else ' for ' + name,
            self.profit_factor,
            self.sharpe,
            self.information,
            100 * self.annualized_return,
            100 * self.annualized_volatility,
            self.drawdown_period,
            100 * self.max_drawdown,
            100 * self.winning_pct
            )

    def plot(self, weights=True, assets=True, portfolio_label='PORTFOLIO', **kwargs):
        """ Plot equity of all assets plus our strategy.
        :param weights: Plot weights as a subplot.
        :param assets: Plot asset prices.
        :return: List of axes.
        """
        res = ListResult([self], [portfolio_label])
        if not weights:
            ax1 = res.plot(assets=assets, **kwargs)
            return [ax1]
        else:
            plt.figure(1)
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            res.plot(assets=assets, ax=ax1, **kwargs)
            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

            # plot weights as lines
            if self.B.values.min() < -0.01:
                self.B.plot(ax=ax2, ylim=(min(0., self.B.values.min()), max(1., self.B.sum(1).max())),
                            legend=False, colormap=plt.get_cmap('jet'))
            else:
                # fix rounding errors near zero
                if self.B.values.min() < 0:
                    B = self.B - self.B.values.min()
                else:
                    B = self.B
                B.plot(ax=ax2, ylim=(0., max(1., B.sum(1).max())),
                       legend=False, colormap=plt.get_cmap('jet'), kind='area', stacked=True)
            plt.ylabel('weights')
            return [ax1, ax2]

    def hedge(self, result=None):
        """ Hedge results with results of other strategy (subtract weights).
        :param result: Other result object. Default is UCRP.
        :return: New AlgoResult object.
        """
        if result is None:
            from algos import CRP
            result = CRP().run(self.X.cumprod())

        return AlgoResult(self.X, self.B - result.B)

    def plot_decomposition(self, **kwargs):
        """ Decompose equity into components of individual assets and plot
        them. Does not take fees into account. """
        ax = self.equity_decomposed.plot(**kwargs)
        return ax

    @property
    def importance(self):
        ws = self.weights.sum()
        return (ws / sum(ws)).order(ascending=False)


class ListResult(list, PickleMixin):
    """ List of AlgoResults. """

    def __init__(self, results=None, names=None):
        results = results if results is not None else []
        names = names if names is not None else []
        super(ListResult, self).__init__(results)
        self.names = names

    def append(self, result, name):
        super(ListResult, self).append(result)
        self.names.append(name)

    def to_dataframe(self):
        """ Calculate equities for all results and return one dataframe. """
        eq = {}
        for result, name in zip(self, self.names):
            eq[name] = result.equity
        return pd.DataFrame(eq)

    def save(self, filename, **kwargs):
        # do not save it with fees
        #self.fee = 0.
        #self.to_dataframe().to_pickle(*args, **kwargs)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        # df = pd.read_pickle(*args, **kwargs)
        # return cls([df[c] for c in df], df.columns)

        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def fee(self):
        return {name: result.fee for result, name in zip(self, self.names)}

    @fee.setter
    def fee(self, value):
        for result in self:
            result.fee = value

    def summary(self):
        return '\n'.join([result.summary(name) for result, name in zip(self, self.names)])

    def plot(self, ucrp=False, bah=False, assets=False, **kwargs):
        """ Plot strategy equity.
        :param ucrp: Add uniform CRP as a benchmark.
        :param bah: Add Buy-And-Hold portfolio as a benchmark.
        :param assets: Add asset prices.
        :param kwargs: Additional arguments for pd.DataFrame.plot
        """
        # NOTE: order of plotting is important because of coloring
        # plot portfolio
        d = self.to_dataframe()
        portfolio = d.copy()
        ax = portfolio.plot(linewidth=3., legend=False, **kwargs)
        kwargs['ax'] = ax

        ax.set_ylabel('Total wealth')

        # plot uniform constant rebalanced portfolio
        if ucrp:
            from algos import CRP
            crp_algo = CRP().run(self[0].X.cumprod())
            crp_algo.fee = self[0].fee
            d['UCRP'] = crp_algo.equity
            d[['UCRP']].plot(**kwargs)

        # add bah
        if bah:
            from algos import BAH
            bah_algo = BAH().run(self[0].X.cumprod())
            bah_algo.fee = self[0].fee
            d['BAH'] = bah_algo.equity
            d[['BAH']].plot(**kwargs)

        # add individual assets
        if assets:
            self[0].asset_equity.plot(colormap=plt.get_cmap('jet'), **kwargs)

        # plot portfolio again to highlight it
        kwargs['color'] = 'blue'
        portfolio.plot(linewidth=3., **kwargs)

        return ax
