# Quant Autoagent — Research Directive

## Meta-Agent Instructions

You are a quantitative research agent. Your job is to iteratively discover and
improve trading strategies by modifying the editable section of `agent.py`,
running the backtest, reading the results, and proposing the next improvement.

The editable section is everything above the `FIXED ADAPTER BOUNDARY` comment.
It contains ONLY two functions — do not add anything else:

- `get_signals(data)` → pd.DataFrame[bool]
- `get_position_sizes(signals, data)` → pd.DataFrame[float]

## Experiment Loop

1. Read the current editable section of `agent.py`
2. Read `results.csv` to understand what has been tried and what failed
3. Propose ONE focused change — explain your reasoning
4. Rewrite the editable section of `agent.py` with the new strategy
5. Run the backtest with a short description of the change
6. Read `last_result.json` for the score
7. If score improved: keep the change and increment your improvement counter. If not: revert `agent.py` to the previous version
8. **If improvement counter is a multiple of 5, you MUST run walk-forward before continuing:**
   - Run `python agent.py --walk-forward --desc "walk-forward check #N"`
   - Read `last_result.json` — check `mean_oos_sharpe` and `pass`
   - If `pass` is `false`: revert to the previous best, increment walk-forward failure counter, and try a more conservative version next
   - If `pass` is `true`: reset walk-forward failure counter and continue
9. Repeat

## Walk-Forward Rule

Every 5 successful improvements, run walk-forward manually.
If mean OOS Sharpe < 0.8, revert to a more conservative version.

## Stopping Criteria

- Score > 1.8 — target achieved
- 50 iterations — time limit
- 3 consecutive walk-forward failures — likely overfit, stop

---

## Research Directive

**Asset class:** US equities (S&P 500 universe)  
**Data:** Daily OHLCV, 2010–2022 in-sample, 2023 holdout (never look at this)  
**Universe:** Top 150 stocks by 30-day average dollar volume

**Target metrics:**

- Sharpe ratio > 1.2 (annualized, after 10bps transaction costs)
- Max drawdown < 25%
- Monthly turnover < 50%
- Long-only (no shorting)

**Score formula:**

```
score = sharpe
      - max(0, (turnover - 0.3) * 0.5)       # penalize high turnover
      - max(0, (|max_drawdown| - 0.20) * 2)  # penalize deep drawdowns
```

**Hypothesis space — explore roughly in this order:**

1. Cross-sectional momentum (rank by N-month return, skip 1 month)
2. Market regime filter (SPY above/below moving average)
3. Volatility-scaled position sizing (inverse vol weighting)
4. Mean reversion on short lookbacks (1-5 day reversal)
5. Trend following on individual stocks (price vs MA)
6. Combining momentum + trend filter
7. Realized vol regime conditioning (reduce exposure in high-vol periods)
8. Volume filters (only trade stocks with above-average volume)
9. Earnings blackout (avoid holding around earnings — proxy via vol spikes)

**What has worked in the academic literature:**

- 12-1 momentum is robust but suffers from momentum crashes in bear markets
- Market regime filter (SPY > 200d MA) materially reduces drawdown
- Volatility scaling improves Sharpe by dampening exposure in volatile periods
- Low turnover is critical — 10bps/trade destroys high-turnover strategies
- Combining signals improves Sharpe vs single-signal strategies

**What tends to overfit:**

- Very short lookbacks on individual signals (< 5 days)
- Too many simultaneous parameter changes
- Strategies that only work in one regime (pure bull-market momentum)

## How to Run

```bash
# Install deps
pip install -e .

# Establish baseline (first run, no changes)
python agent.py --in-sample --desc "baseline: 12-1 momentum + SPY 200d regime"

# After editing the editable section, run a backtest
python agent.py --in-sample --desc "added inverse vol weighting to position sizes"

# Run walk-forward validation (every 5 improvements)
python agent.py --walk-forward --desc "walk-forward check after vol scaling"

# Final holdout eval (only once, at the end)
python agent.py --holdout --desc "final holdout: best strategy"
```
