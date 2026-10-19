---
name: add-algorithm
description: Add a new online portfolio selection algorithm to universal-portfolios, or finish/review a PR that adds one. Use when someone asks to implement an algorithm from a paper (e.g. "add AdaHedge", "implement SSPO"), to port one from another library, or to review or finish a contributed algorithm PR. Covers the Algo contract and its timing, registration, tests that pin down the paper's claims, checking the implementation against the paper, and the math doc that sits next to the code.
---

# Adding an algorithm

The library's value is that every algorithm is a faithful implementation of its paper and runs under the same `Algo` contract, so results can be compared. Most bugs in contributed algorithms come from the contract (weights computed a period late, state that skips periods) or from quietly departing from the paper (extra parameters, clipping, a different start), not from the formula itself. So read the paper first, write the code to match it, then prove that it does.

## Read the paper first

Find the algorithm's definition: pseudocode, update rule, initial values, and how ties and limits (η = ∞, zero history) are handled. Note what the paper proves (regret bound, special cases equal to known strategies), because those become the tests. When the paper has reference code, follow it. When a contribution adds safeguards the paper doesn't have (epsilons, learning-rate clipping, loss clipping), check whether the paper's own definition already handles the case. AdaHedge's PR approximated η₁ = ∞ with three tuning constants the paper doesn't need.

If the paper is behind a bot challenge or paywall, ask the user for a copy rather than working from a summary.

## The `Algo` contract

Read `universal/algo.py`. The parts that matter:

- **Timing.** `step(x, last_b, history)` is called after period t with that period's returns `x`. It returns the weights held during period t+1, and `AlgoResult` applies `B[t]` to `X[t]` with no shift. Use `x` before computing the new weights. The classic bug is computing weights from state and then adding `x`, which returns weights one period stale. `test_bias` doesn't catch that, because stale weights don't look ahead.
- **State.** Initialise anything kept between steps in `init_step(X)`, not `__init__`. `next_weights` replays the history through `init_step` + `step`, and with `frequency > 1` each `step` gets the return accumulated since its previous call. Both only work if all state lives there.
- **Inputs.** `PRICE_TYPE` picks what `x` and `history` are (`"ratio"`, `"log"` or `"raw"`). Pick the one the paper's formula uses, so the code reads like the paper. `REPLACE_MISSING = True` fills missing prices with "no change". Without it, NaNs reach `step`, which matters for assets listed late.
- **Vectorised algorithms** can override `weights(X)` instead of `step`. They lose `next_weights`, so add them to `NO_STEP` in `tests/test_algos.py`.
- `init_weights` defaults to all zeros (cash); most algorithms want uniform weights.

Match the surrounding files: a docstring with what the algorithm does in a sentence or two plus its reference, `__init__` parameters documented with `:param`, and an `if __name__ == "__main__": tools.quickrun(...)` block.

## Register it

Add the import and the name to `__all__` in `universal/algos/__init__.py`. The shared tests iterate over `__all__`, so this is also what puts it under test. Add it to the README's algorithm table in the column that fits.

## Tests

`tests/test_algos.py` already runs every registered algorithm through the shared invariants:
- no look-ahead;
- finite, non-negative weights summing to at most 1;
- `next_weights` matching `run`.

If the new algorithm legitimately breaks one of these, add it to the matching exclusion set (`HINDSIGHT`, `LEVERAGED`, `NO_STEP`), and only for a reason that is part of its design. If it fails for any other reason, that's a bug.

Algorithm-specific tests go in `tests/test_<name>.py` and should pin down what the paper says, not snapshot today's output:
- **Limits that equal known algorithms**, e.g. a learning rate at which it becomes BAH, uniform CRP or follow-the-leader.
- **The regret bound** on a case built to stress it, such as a leader that flips every period. Use the theorem's full bound, constants included. If you assert a tighter threshold, say it's empirical.
- **Small hand-checkable cases** where the paper's update can be computed by hand.
- **Edge cases the paper's assumptions exclude**, where the code still has to behave: a single asset, an asset listed late, a constant price, a duplicated asset.

Check that the tests can fail. Revert the fix or plant a plausible bug, such as the one-period lag, and confirm at least one test goes red. A test that passes on both versions proves nothing about the change.

## Verify against the paper

For anything beyond a one-line update, get an independent check. Have a subagent transcribe the paper's algorithm onto a plain loss or return matrix without reading the library code. It runs that transcription on the same inputs the library sees (random data, a bundled dataset, NaNs, ties, a single asset) and compares the weights numerically with `Algo().run(S).B`. It also checks the paper's bound empirically. Ask for numbers and the quoted theorem text, not a verdict.

Quote theorem and equation numbers only after checking them against the paper's own text.

## Document the math

Write `universal/algos/<name>.md` next to the code and point to it from the docstring. Cover:
- the setup and notation;
- the update rule as the paper states it;
- what the parameters do, including any limits that equal known algorithms;
- the guarantees, with theorem numbers;
- what they mean for portfolio wealth;
- a table mapping symbols to code;
- the step timing;
- known limitations.

GitHub renders math: use `` $`...`$ `` for inline math and ```` ```math ```` blocks for display. Plain `$...$` can break when the LaTeX contains `_` or `*`, which Markdown reads as emphasis. After pushing, look at the rendered file on GitHub. A formula that didn't render shows up as raw LaTeX.

## Before opening the PR

- Run it against the benchmarks (`CRP`, `BAH`, `BestSoFar`, and the closest related algorithm) on a few bundled datasets (`tools.dataset("nyse_o")`, `"djia"`, `"tse"`, `"sp500"`). Say plainly in the docstring and the doc what it is good for. A follow-the-winner algorithm that trails uniform CRP on raw stocks isn't wrong, but users should know.
- Run `make check`. It formats, lints and typechecks, and CI runs `make test` on Python 3.11–3.14.
- In the PR description, list where the code departs from the paper, if anywhere, and why.
