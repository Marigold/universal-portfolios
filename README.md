# universal-portfolios

The purpose of this package is to put together different online portfolio selection algorithms and provide unified tools for their analysis. If you do not know what online portfolio is, look at [Ernest Chan blog](http://epchan.blogspot.cz/2007/01/universal-portfolios.html), [CASTrader](http://www.castrader.com/2006/11/universal_portf.html) or a recent [survey by Bin Li and Steven C. H. Hoi](http://arxiv.org/abs/1212.2129).

In short, the purpose of online portfolio is to _choose portfolio weights in every period to maximize its final wealth_. Examples of such portfolios could be [Markowitz portfolio](http://en.wikipedia.org/wiki/Modern_portfolio_theory) or [Universal portfolio](http://en.wikipedia.org/wiki/Universal_portfolio_algorithm). Currently there is an active research in the are of online portfolios and even though its results are mostly theoretic, algorithms for practical use starts to appear.

Several algorithms from the literature are currently implemented, based on the available literature and my understanding. Contributions or corrections are more than welcomed.

## Resources

There is an [IPython notebook](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/On-line%20portfolios.ipynb) explaining the basic use of the library. Paul Perry followed up on this and made a [comparison of all algorithms](http://nbviewer.ipython.org/github/paulperry/quant/blob/master/OLPS_Comparison.ipynb) on more recent ETF datasets. Also see the most recent [notebook about modern portfolio theory](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/modern-portfolio-theory.ipynb). There's also an interesting discussion about this on [Quantopian](https://www.quantopian.com/posts/comparing-olps-algorithms-olmar-up-et-al-dot-on-etfs#553a704e7c9031e3c70003a9).

The original authors of some of the algorithms recently published their own implementation on github - [On-Line Portfolio Selection Toolbox](https://github.com/OLPS/OLPS) in MATLAB.

If you are more into R or just looking for a good resource about Universal Portfolios, check out blog and package [logopt](http://optimallog.blogspot.cz/) by Marc Delvaux.

If you don't want to install the package locally, you can run both notebooks with Binder - [modern-portfolio-theory.ipynb ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marigold/universal-portfolios/master?filepath=modern-portfolio-theory.ipynb) or [On-line portfolios.ipynb ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marigold/universal-portfolios/master?filepath=On-line%20portfolios.ipynb)

## Installation

```
pip install universal-portfolios
```

## Development

It uses poetry to manage dependencies. Run `poetry install` to install virtual environment and then `poetry shell` to launch it.

Exporting dependencies to `requirements.txt` file is done via

```
poetry export --without-hashes -f requirements.txt > requirements.txt
poetry export --dev --without-hashes -f requirements.txt > test-requirements.txt
```

this is needed for mybinder.org.

## Publishing to PyPi

```
poetry publish --build
```

## Tests

```
poetry run python -m pytest --capture=no --ff -x tests/
```
