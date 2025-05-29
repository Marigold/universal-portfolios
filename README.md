# Universal Portfolios

![PyPi Version](https://img.shields.io/pypi/v/universal-portfolios?style=flat-square)
![PyPi License](https://img.shields.io/pypi/l/universal-portfolios?style=flat-square)
![PyPi Downloads](https://img.shields.io/pypi/dm/universal-portfolios?style=flat-square)

![Open PRs](https://img.shields.io/github/issues-pr-raw/Marigold/universal-portfolios?style=flat-square)
![Contributors](https://img.shields.io/badge/contributors-9-orange.svg?style=flat-square)
![Repo size](https://img.shields.io/github/repo-size/Marigold/universal-portfolios?style=flat-square)

The purpose of this Python package is to bring together different Online Portfolio Selection (OLPS) algorithms and provide unified tools for their analysis.

In short, the purpose of OLPS is to _choose portfolio weights in every period to maximize final wealth_. Examples of such portfolios include the [Markowitz portfolio](http://en.wikipedia.org/wiki/Modern_portfolio_theory) or the [Universal portfolio](http://en.wikipedia.org/wiki/Universal_portfolio_algorithm). There is currently active research in the area of online portfolios, and even though the results are mostly theoretical, algorithms for practical use are starting to appear.

Several state-of-the-art algorithms are implemented, based on my understanding of the available literature. Contributions or corrections are more than welcome.

## Outline of this package

* `examples` contains two Python Notebooks:
   - [Online Portfolios](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/On-line%20portfolios.ipynb): explains the basic use of the library. Script sequence, various options, method arguments, and a strategy template to get you started.
   - [Modern Portfolio Theory](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/modern-portfolio-theory.ipynb): goes deeper into the OLPS principle and the tools developed in this library to approach it.

* `universal.data` contains various datasets to help you in your journey.

* `universal.algos` hosts the implementations of various OLPS algorithms from the literature:

<div align="center">

| Benchmarks | Follow the winner | Follow the loser | Pattern matching | Other |
|---|---|---|---|---|
| __[BAH](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bah.py)__ | __[Universal Portfolios](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/up.py)__ | __[Anticorr](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/anticor.py)__ | __[BNN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bnn.py)__ | __[Markowitz](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_markowitz.py)__ |
| __[CRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/crp.py)__ | __[Exponential Gradient](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/eg.py)__ | __[PAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/pamr.py)__ | __[CORN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/corn.py)__ | __[Kelly](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/kelly.py)__ |
| __[BCRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bcrp.py)__ || __[OLMAR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/olmar.py)__ || __[Best so far](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_so_far.py)__ |
| __[DCRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/dynamic_crp.py)__ || __[RMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/rmr.py)__ || __[ONS](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ons.py)__ |
||| __[CWMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/cwmr.py)__ || __[MPT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/mpt.py)__ |
||| __[WMAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/wmamr.py)__ |||
||| __[RPRT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/rprt.py)__ |||

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
