# Universal Portfolios


![PyPi Version](https://img.shields.io/pypi/v/universal-portfolios?style=flat-square)
![PyPi License](https://img.shields.io/pypi/l/universal-portfolios?style=flat-square)
![PyPi Downloads](https://img.shields.io/pypi/dm/universal-portfolios?style=flat-square)
![Open PRs](https://img.shields.io/github/issues-pr-raw/Marigold/universal-portfolios?style=flat-square)
![Contributors](https://img.shields.io/badge/contributors-9-orange.svg?style=flat-square)
![Repo size](https://img.shields.io/github/repo-size/Marigold/universal-portfolios?style=flat-square)

The purpose of this Python package is to put together different Online Portfolio Selection (OLPS) algorithms and provide unified tools for their analysis.


In short, the purpose of OLPS is to _choose portfolio weights in every period to maximize its final wealth_. Examples of such portfolios could be the [Markowitz portfolio](http://en.wikipedia.org/wiki/Modern_portfolio_theory) or the [Universal portfolio](http://en.wikipedia.org/wiki/Universal_portfolio_algorithm). Currently there is an active research in the are of online portfolios and even though its results are mostly theoretic, algorithms for practical use starts to appear.

Several algorithms from the literature are currently implemented, based on the available literature and my understanding. Contributions or corrections are more than welcomed.

## Outline of this package

* In the `examples` folder are two Python Notebooks: 
   - [Online Portfolios](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/On-line%20portfolios.ipynb) : explains the basic use of the library.
   - [Modern Portfolio Theory](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/modern-portfolio-theory.ipynb) : ...

* `universal.data` contains various datasets to help you in your journey

* `universal.algos` hosts the implementations of various OLPS algorithms from the litterature :
<!--
 - [Anticorr](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/anticor.py)
 - [BAH](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bah.py)
 - [BCRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bcrp.py)
 - [Markovitz](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_markowitz.py)
 - [Best so far](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_so_far.py)
 - [BNN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bnn.py)
 - [CORN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/corn.py)
 - [CRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/crp.py)
 - [Dynamic CRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/dynamic_crp.py)
 - [CWMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/cwmr.py)
 - [Exponential Gradient](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/eg.py)
 - [Kelly](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/kelly.py)
 - [MPT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/mpt.py)
 - [OLMAR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/olmar.py)
 - [ONS](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ons.py)
 - [PAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/pamr.py)
 - [RMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/rmr.py)
 - [SICE](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/single_index_covariance_estimator.py)
 - [Universal Portfolios](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/up.py)
 - [WMAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/wmamr.py)
-->

| | | | |
|---|---|---|---|
| [Anticorr](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/anticor.py) | [BNN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bnn.py) | [Exponential Gradient](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/eg.py) | [PAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/pamr.py) |
| [BAH](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bah.py) | [CORN](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/corn.py) | [Kelly](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/kelly.py) | [RMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/rmr.py) |
| [BCRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/bcrp.py) | [CRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/crp.py) | [MPT](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/mpt.py) | [SICE](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/single_index_covariance_estimator.py) |
| [Markovitz](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_markowitz.py) | [Dynamic CRP](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/dynamic_crp.py) | [OLMAR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/olmar.py) | [Universal Portfolios](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/up.py) |
| [Best so far](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/best_so_far.py) | [CWMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/cwmr.py) | [ONS](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ons.py) | [WMAMR](https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/wmamr.py) |

* `universal.algo` provides a general class inherited by all the individual algos' subclasses. Algo computes the weights at every timestep.

* `universal.result` computes the portfolio wealth from the weights and various metrics on the strategy's performance.

 # **Quick Start**
 
```python
from universal import tools
from universal.algos import CRP

if __name__ == '__main__':
  # Run CRP on a computed-generated portfolio of 3 stocks and plot the results
  tools.quickrun(CRP())

```


## Additional Resources

If you do not know what online portfolio is, look at [Ernest Chan blog](http://epchan.blogspot.cz/2007/01/universal-portfolios.html), [CASTrader](http://www.castrader.com/2006/11/universal_portf.html) or a recent [survey by Bin Li and Steven C. H. Hoi](http://arxiv.org/abs/1212.2129).

Paul Perry followed up on this and made a [comparison of all algorithms](http://nbviewer.ipython.org/github/paulperry/quant/blob/master/OLPS_Comparison.ipynb) on more recent ETF datasets. The original authors of some of the algorithms recently published their own implementation on github - [On-Line Portfolio Selection Toolbox](https://github.com/OLPS/OLPS) in MATLAB.

If you are more into R or just looking for a good resource about Universal Portfolios, check out blog and package [logopt](http://optimallog.blogspot.cz/) by Marc Delvaux.

Note : If you don't want to install the package locally, you can run both notebooks with Binder - [modern-portfolio-theory.ipynb ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marigold/universal-portfolios/master?filepath=modern-portfolio-theory.ipynb) or [On-line portfolios.ipynb ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marigold/universal-portfolios/master?filepath=On-line%20portfolios.ipynb)

## Installation

```
pip install universal-portfolios
```

## Development

It uses poetry to manage dependencies. Run `poetry install` to install virtual environment and then `poetry shell` to launch it.

Exporting dependencies to `requirements.txt` file is needed for mybinder.org. It is done via

```
poetry export --without-hashes -f requirements.txt > requirements.txt
poetry export --dev --without-hashes -f requirements.txt > test-requirements.txt
```

## Running Tests

```
poetry run python -m pytest --capture=no --ff -x tests/
```

## Contributors

Creator : [Marigold](https://github.com/Marigold)

_Thank you for your contributions!_

[Alexander Myltsev](https://github.com/alexander-myltsev) | [angonyfox](https://github.com/angonyfox) | [booxter](https://github.com/booxter) | [dexhunter](https://github.com/dexhunter) | [DrPaprikaa](https://github.com/DrPaprikaa) | [paulorodriguesxv](https://github.com/paulorodriguesxv) | [stergnator](https://github.com/stergnator) | [Xander Dunn](https://github.com/xanderdunn)

## Disclaimer

This software is for educational purposes only and is far from any production environment. Do not risk money which you are afraid to lose.
Use the software at your own risk. The authors assume no responsibility for your trading results. 
