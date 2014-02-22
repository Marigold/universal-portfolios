
universal-portfolios
===========

The purpose of this package is to put together different online portfolio selection algorithms and provide unified tools for their analysis. If you do not know what online portfolio is, look at [Ernest Chan blog](http://epchan.blogspot.cz/2007/01/universal-portfolios.html), [CASTrader](http://www.castrader.com/2006/11/universal_portf.html) or a recent [survey by Bin Li and Steven C. H. Hoi](http://arxiv.org/abs/1212.2129). 

In short, the purpose of online portfolio is to *choose portfolio weights in every period to maximize its final wealth*. Examples of such portfolios could be [Markowitz portfolio](http://en.wikipedia.org/wiki/Modern_portfolio_theory) or [Universal portfolio](http://en.wikipedia.org/wiki/Universal_portfolio_algorithm). Currently there is an active research in the are of online portfolios and even though its results are mostly theoretic, algorithms for practical use starts to appear.

Several algorithms from the literature are currently implemented, based on the literature and my understanding. Contributions or corrections are more than welcomed.


## Installation


####Dependencies

The usual scientific libraries: numpy, scipy, pandas, matplotlib and cvxopt. All of them should be included in [Anaconda](https://store.continuum.io/cshop/anaconda/).

#### Installing
      
        pip install universal-portfolios


## Introduction

There is an [IPython notebook](http://nbviewer.ipython.org/github/Marigold/universal-portfolios/blob/master/On-line%20portfolios.ipynb) explaining the basic use of the library.
