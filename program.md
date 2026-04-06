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

1. Read the current editable section of `agent.py`.

2. Read `results.csv` to understand what has been tried and what failed.
   - Inspect the INEFFICIENCY field of every past `--desc` entry.
   - Do not repeat an approach unless you have a genuinely new economic
     justification. Parameter sweeps (e.g. "try 93rd pct instead of 95th")
     with no new rationale are forbidden.

3. Propose ONE focused change. Before writing any code, write a rationale
   using this exact format:

   **CHANGE:** One sentence on what you are changing.  
   **WHY:** One sentence on why it should improve the score — name a specific
   mechanism. "To see if it helps" is not valid.

   If you cannot write a convincing WHY, do not run the experiment.
   Move to the next hypothesis in the hypothesis space instead.

4. Rewrite the editable section of `agent.py` with the new strategy.
   Change ONE thing at a time. If more than ~10 lines change, you are likely
   making a compound change — split it into separate experiments.

5. Run the backtest. The `--desc` argument is mandatory and must follow this
   structure exactly (keep each sentence under 100 characters):

   ```
   python agent.py --in-sample --desc "CHANGE: <what you are changing>. WHY: <why it should improve the score>."
   ```

   Do NOT run the experiment if you cannot write a honest WHY sentence.
   "WHY: to see if it helps" is not valid. A WHY must name a specific
   mechanism — if you cannot name one, the experiment should not run.

6. Read `last_result.json` for the score.

7. If score improved: keep the change and increment your improvement counter.
   If not: revert `agent.py` to the previous version. Do not try a minor
   variation of the same idea without a new economic justification — a failed
   hypothesis is a failed hypothesis.

8. **If improvement counter is a multiple of 5, you MUST run walk-forward
   before continuing:**
   - Run: `python agent.py --walk-forward --desc "CHANGE: <same as last kept change>. WF-CHECK: #N"`
   - Read `last_result.json` — check `mean_oos_sharpe` and `pass`.
   - If `pass` is `false`: revert to the previous best. Ask whether the
     underlying inefficiency hypothesis is still convincing given the OOS
     evidence. If the same hypothesis has failed walk-forward twice, abandon
     it entirely and move to a new one from the hypothesis space below.
   - If `pass` is `true`: reset walk-forward failure counter and continue.

9. Repeat until a stopping criterion is met.

## Walk-Forward Rule

Every 3 successful improvements, run walk-forward manually.
If mean OOS Sharpe < 0.8, revert to a more conservative version.
If the same hypothesis fails walk-forward twice in a row, it is overfit —
discard it and start from a new hypothesis.

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
