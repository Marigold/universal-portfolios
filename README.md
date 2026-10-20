# Universal Portfolios

![PyPi Version](https://img.shields.io/pypi/v/universal-portfolios?style=flat-square)
![PyPi License](https://img.shields.io/pypi/l/universal-portfolios?style=flat-square)
![PyPi Downloads](https://img.shields.io/pypi/dm/universal-portfolios?style=flat-square)

![Open PRs](https://img.shields.io/github/issues-pr-raw/Marigold/universal-portfolios?style=flat-square)
![Contributors](https://img.shields.io/badge/contributors-9-orange.svg?style=flat-square)
![Repo size](https://img.shields.io/github/repo-size/Marigold/universal-portfolios?style=flat-square)

This Python package brings together Online Portfolio Selection (OLPS) algorithms from the literature, implemented faithfully to their papers and run under one interface, so that they can be compared and their claims checked.

In short, OLPS algorithms _choose portfolio weights in every period to maximize final wealth_. Examples include the [Markowitz portfolio](http://en.wikipedia.org/wiki/Modern_portfolio_theory) and the [Universal portfolio](http://en.wikipedia.org/wiki/Universal_portfolio_algorithm). Many papers in this field report spectacular backtests. The point of this package is to make it easy to see how much of that survives a closer look. Contributions and corrections are more than welcome.

## ⚠️ Read this before trusting any backtest

The results you get from these algorithms on the bundled datasets say very little about how they would do with real money.

- **The datasets only contain survivors.** NYSE(O), NYSE(N), DJIA, SP500, TSE and MSCI are the standard benchmarks of the OLPS literature. They include only stocks that traded every day of the period: none went bankrupt or was delisted. Some are also picked in hindsight. SP500 is the 25 stocks with the largest market cap _at the end_ of the period, and NYSE(N) keeps 23 of NYSE(O)'s 36 stocks. The [Anticor paper](https://jair.org/index.php/jair/article/download/10380/24854/19216) (p. 592), [CWMR](http://proceedings.mlr.press/v15/li11b/li11b.pdf) (§5.3) and [OLMAR](https://arxiv.org/abs/1206.4626) (§5.5) acknowledge this. Many later papers, such as [PPT](https://ieeexplore.ieee.org/document/7942104) and [SSPO](https://jmlr.org/papers/v19/17-558.html), don't mention it.
- **Survivor-only data hides the main risk of "follow the loser" algorithms.** OLMAR, PAMR, PPT, SSPO and the like move money into stocks that just fell. In survivor-only data, every stock that falls recovers. [`examples/survivorship_bias.ipynb`](examples/survivorship_bias.ipynb) simulates this on random walks. Survivorship alone doesn't make these algorithms beat uniform CRP. But once stocks can go bankrupt, they keep buying into the collapsing ones and end with as little as 4% of CRP's wealth, while buy-and-hold isn't hurt.
- **Transaction costs.** Most headline numbers assume rebalancing for free at the daily close. The mean-reversion algorithms trade most of their wealth every day. [PPT](universal/algos/ppt.md), for example, falls behind uniform CRP at fees of 0.14–0.6% per unit traded on the bundled datasets, and [SSPO](universal/algos/sspo.md) at 0.5% round trip ends below CRP on SP500. Set `result.fee` before believing any number.
- **Parameters are often tuned on the same data.** PPT's step size and all five of SSPO's parameters were chosen by final wealth on the same five datasets the papers then report, with no out-of-sample period.
- **Independent re-tests are discouraging.** On recent S&P 500 stocks, [Moon et al. (2019)](https://arxiv.org/abs/1909.04327) find that OLMAR and PAMR lose almost everything at 0.25% costs.

Use this package to study the algorithms and their failure modes, not to pick a trading strategy.

## Outline of this package

* Python notebooks:
   - [Online Portfolios](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/On-line%20portfolios.ipynb): explains the basic use of the library. Script sequence, various options, method arguments, and a strategy template to get you started.
   - [Modern Portfolio Theory](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/modern-portfolio-theory.ipynb): goes deeper into the OLPS principle and the tools developed in this library to approach it.
   - [Survivorship bias](examples/survivorship_bias.ipynb): what survivor-only datasets hide about mean-reversion algorithms, on simulated data.

* `universal.data` contains various datasets to help you in your journey.

* `universal.algos` hosts the implementations of various OLPS algorithms from the literature:

<div align="center">

| Benchmarks | Follow the winner | Follow the loser | Pattern matching | Other |
|---|---|---|---|---|
| __[BAH](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bah.py)__ | __[Universal Portfolios](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/up.py)__ | __[Anticorr](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/anticor.py)__ | __[BNN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bnn.py)__ | __[Markowitz](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_markowitz.py)__ |
| __[CRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/crp.py)__ | __[Exponential Gradient](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/eg.py)__ | __[PAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/pamr.py)__ | __[CORN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/corn.py)__ | __[Kelly](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/kelly.py)__ |
| __[BCRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bcrp.py)__ | __[Soft-Bayes](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/soft_bayes.py)__ | __[OLMAR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/olmar.py)__ || __[Best so far](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_so_far.py)__ |
| __[DCRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/dynamic_crp.py)__ | __[AdaHedge](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/adahedge.py)__ | __[RMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/rmr.py)__ || __[ONS](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ons.py)__ |
||| __[CWMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/cwmr.py)__ || __[MPT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/mpt.py)__ |
||| __[WMAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/wmamr.py)__ |||
||| __[RPRT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/rprt.py)__ |||
||| __[PPT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ppt.py)__ |||
||| __[SSPO](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/sspo.py)__ |||

</div>

* `universal.algo` provides a general class inherited by all the individual algos' subclasses. Algo computes the weights at every timestep.

* `universal.result` computes the portfolio wealth from the weights and various metrics on the strategy's performance.

## Quick Start

```python
from universal import tools
from universal.algos import CRP

if __name__ == '__main__':
  # Run CRP on a computer-generated portfolio of 3 stocks and plot the results
  tools.quickrun(CRP())
```

## Additional Resources

If you do not know what an online portfolio is, look at [Ernest Chan's blog](http://epchan.blogspot.cz/2007/01/universal-portfolios.html), [CASTrader](http://www.castrader.com/2006/11/universal_portf.html), or a recent [survey by Bin Li and Steven C. H. Hoi](http://arxiv.org/abs/1212.2129).

Paul Perry followed up on this and made a [comparison of all algorithms](http://nbviewer.ipython.org/github/paulperry/quant/blob/master/OLPS_Comparison.ipynb) on more recent ETF datasets. The original authors of some of the algorithms recently published their own implementation on GitHub - [Online Portfolio Selection Toolbox](https://github.com/OLPS/OLPS) in MATLAB.

If you are more into R or just looking for a good resource about Universal Portfolios, check out the blog and package [logopt](http://optimallog.blogspot.cz/) by Marc Delvaux.

Note: If you don't want to install the package locally, you can run both notebooks with Binder - [modern-portfolio-theory.ipynb ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marigold/universal-portfolios/master?filepath=modern-portfolio-theory.ipynb) or [On-line portfolios.ipynb ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marigold/universal-portfolios/master?filepath=On-line%20portfolios.ipynb)

## Installation

```
pip install universal-portfolios
```

## Development

Set up .venv, install dependencies and run tests with:

```
make test
```

## Contributors

Creator: [Marigold](https://github.com/Marigold)

_Thank you for your contributions!_

[Alexander Myltsev](https://github.com/alexander-myltsev) | [angonyfox](https://github.com/angonyfox) | [booxter](https://github.com/booxter) | [dexhunter](https://github.com/dexhunter) | [DrPaprikaa](https://github.com/DrPaprikaa) | [paulorodriguesxv](https://github.com/paulorodriguesxv) | [stergnator](https://github.com/stergnator) | [Xander Dunn](https://github.com/xanderdunn)

## Disclaimer

This software is for educational purposes only and is far from any production environment. Do not risk money you are afraid to lose.
Use the software at your own risk. The authors assume no responsibility for your trading results.
