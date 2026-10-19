# AdaHedge

AdaHedge is the Hedge (exponential weights) algorithm with a learning rate that tunes itself from the data. [`adahedge.py`](adahedge.py) applies it to portfolio selection by treating every asset as an expert whose loss is its negative log return. It has no parameters.

Reference: S. de Rooij, T. van Erven, P. D. Grünwald, W. M. Koolen, [Follow the Leader If You Can, Hedge If You Must](https://arxiv.org/abs/1301.0534), JMLR 15 (2014). Equation and theorem numbers below refer to that paper.

## Setup

There are $`K`$ assets and periods $`t = 1, \dots, T`$. Asset $`k`$ has price relative $`x_{t,k} = p_{t,k} / p_{t-1,k}`$ and loss

```math
\ell_{t,k} = -\ln x_{t,k}, \qquad L_{t,k} = \sum_{s=1}^{t} \ell_{s,k}, \qquad L^*_t = \min_k L_{t,k},
```

so $`L_{t,k}`$ is the cumulative loss of asset $`k`$, $`e^{-L_{t,k}} = p_{t,k} / p_{0,k}`$ is its growth so far, and $`L^*_t`$ is the loss of the best asset so far.

## Hedge with a fixed learning rate

With learning rate $`\eta > 0`$, the weights held in period $`t`$ are

```math
w_{t,k} = \frac{e^{-\eta L_{t-1,k}}}{\sum_{j=1}^{K} e^{-\eta L_{t-1,j}}} \qquad \text{(eq. 2, uniform prior)}.
```

$`\eta`$ interpolates between three classic strategies:

| $`\eta`$ | weights | strategy |
|---|---|---|
| $`\eta \to 0`$ | $`w_{t,k} = 1/K`$ | uniform constant rebalanced portfolio ([CRP](crp.py)) |
| $`\eta = 1`$ | $`w_{t,k} \propto p_{t-1,k} / p_{0,k}`$ | buy and hold ([BAH](bah.py)) |
| $`\eta \to \infty`$ | uniform over $`\{k : L_{t-1,k} = L^*_{t-1}\}`$ | follow the leader, all in on the best asset so far ([BestSoFar](best_so_far.py)) |

Following the leader is ideal when the leader rarely changes, and terrible when it keeps changing. A small $`\eta`$ is safe but slow to commit. AdaHedge picks $`\eta`$ from how costly hedging has been so far.

## Hedge loss, mix loss and the mixability gap

In period $`t`$ the hedge loss is the weighted average loss, and the mix loss is a log-sum-exp version of it (eq. 3):

```math
h_t = w_t \cdot \ell_t, \qquad m_t = -\frac{1}{\eta} \ln\left(w_t \cdot e^{-\eta \ell_t}\right).
```

For $`\eta = \infty`$ the mix loss becomes $`m_t = L^*_t - L^*_{t-1}`$. The difference

```math
\delta_t = h_t - m_t \ge 0, \qquad \Delta_T = \sum_{t=1}^{T} \delta_t
```

is the mixability gap; it is nonnegative by Jensen's inequality (Lemma 1). The regret to the best asset splits into two parts (eq. 4):

```math
R_T = \sum_{t=1}^{T} h_t - L^*_T = \underbrace{\left(\sum_{t=1}^{T} m_t - L^*_T\right)}_{\le \ln K / \eta} + \Delta_T .
```

A large $`\eta`$ makes the first part small but lets the gap $`\Delta_T`$ grow. A small $`\eta`$ does the opposite.

## AdaHedge

AdaHedge balances the two parts by setting the learning rate from the gap observed so far (eq. 7):

```math
\eta_t = \frac{\ln K}{\Delta_{t-1}}, \qquad \Delta_0 = 0 \;\Rightarrow\; \eta_1 = \infty .
```

It starts by following the leader, and lowers $`\eta`$ only as following the leader turns out to be costly. Each period:

1. Set $`\eta_t = \ln K / \Delta_{t-1}`$, or $`\infty`$ while $`\Delta_{t-1} = 0`$.
2. Hold $`w_t \propto e^{-\eta_t L_{t-1}}`$.
3. Observe $`\ell_t`$ and update $`L_t = L_{t-1} + \ell_t`$.
4. Add the gap of this period, computed with the same $`\eta_t`$: $`\Delta_t = \Delta_{t-1} + h_t - m_t`$.

## Guarantees

The regret is at most twice the gap (Lemma 3), $`R_T \le 2 \Delta_T`$. AdaHedge is invariant to translating and rescaling the losses in each period (Theorem 16), so its bound holds for any real losses (Corollary 17). With

```math
\mu_t = \min_k \ell_{t,k}, \qquad \sigma = \max_t \max_k \left(\ell_{t,k} - \mu_t\right), \qquad N^*_T = L^*_T - \sum_{t=1}^{T} \mu_t ,
```

it guarantees

```math
R_T \le 2 \sqrt{\frac{N^*_T (\sigma T - N^*_T)}{T} \ln K} + \sigma \left(\tfrac{16}{3} \ln K + 2\right) \le \sigma \sqrt{T \ln K} + \sigma \left(\tfrac{16}{3} \ln K + 2\right).
```

So the regret grows at most like $`\sqrt{T}`$ in the worst case, and is much smaller when the best asset is also close to the best one in almost every period ($`N^*_T`$ small). Follow the leader alone can have regret linear in $`T`$.

In portfolio terms, holding $`w_t`$ multiplies wealth by $`w_t \cdot x_t`$. By Jensen's inequality its log loss is at most the hedge loss, $`-\ln(w_t \cdot x_t) \le w_t \cdot \ell_t = h_t`$, so

```math
\ln W_T \ \ge\ \ln W^*_T - R_T ,
```

where $`W_T`$ is the final wealth of AdaHedge (rebalanced every period) and $`W^*_T`$ that of the best single asset. The guarantee is against the best asset, not against the best rebalancing strategy, which is why AdaHedge on raw stocks trails uniform CRP on all the bundled datasets we tried (nyse_o, djia, tse, sp500).

## Mixing strategies

The experts don't have to be assets. Run on a `ListResult`, AdaHedge treats each strategy's equity curve as an asset and stays close to the best strategy, without knowing in advance which one it will be:

```python
from universal import algos
from universal.result import ListResult

experts = [algos.CRP(), algos.OLMAR(), algos.PAMR(), algos.ONS()]
results = ListResult([a.run(S) for a in experts], ["CRP", "OLMAR", "PAMR", "ONS"])
mixed = algos.AdaHedge().run(results)
```

## How the code maps to this

| math | code |
|---|---|
| $`\ell_t`$ | `-x`; with `PRICE_TYPE = "log"`, `step` gets log returns |
| $`L_t`$ | `self.L` |
| $`\Delta_t`$ | `self.Delta` |
| $`\eta_t`$ | `self._eta()` |
| $`w_t`$ and $`M`$ below | `self._mix(eta, L)` |

As in the paper's reference implementation (Figure 1), the mix loss is computed as a difference of the cumulative mix loss

```math
M(\eta, L) = L^* - \frac{1}{\eta} \ln\left(\frac{1}{K} \sum_{k=1}^{K} e^{-\eta (L_k - L^*)}\right), \qquad m_t = M(\eta_t, L_t) - M(\eta_t, L_{t-1}),
```

which subtracts the minimum before exponentiating so that nothing overflows, and reduces to $`M = L^*`$ for $`\eta = \infty`$. `max(0, ...)` on the gap only absorbs rounding.

`step(x)` is called after period $`t`$ with that period's returns. It first closes period $`t`$, using the weights held during it, recomputed from $`L_{t-1}`$ and $`\Delta_{t-1}`$. It then returns $`w_{t+1}`$, which the framework holds during period $`t+1`$.

## Limitations

- With a single asset there is nothing to hedge, and the code keeps $`\eta = \infty`$.
- A price that drops to zero gives an infinite loss, and every later weight is NaN. The paper assumes bounded losses.
- Losses that are equal in exact arithmetic can differ in their last digits after the price conversion. While following the leader, that creates a tiny gap and a huge but finite $`\eta`$, so the weights can briefly be close to, rather than exactly, uniform over the tied leaders. The paper's reference code behaves the same way.
