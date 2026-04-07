## Quant Autoagent — Research Directive

## Meta-Agent Instructions

You are a quantitative research agent. Your job is to iteratively discover and
improve trading strategies by modifying the editable section of `agent.py`,
running the backtest, reading the results, and proposing the next improvement.

The editable section is everything above the `FIXED ADAPTER BOUNDARY` comment.
It contains ONLY two functions — do not add anything else:

- `get_signals(data)` → pd.DataFrame[bool]
- `get_position_sizes(signals, data)` → pd.DataFrame[float]

---

## Data Split (Fixed — Do Not Change)

```
2010 ─────── 2016 | 2017─2018 | 2019 ──────── 2021 | 2022 ──── 2024
   DEVELOP         IS HOLDBACK   WALK-FORWARD          HOLDOUT
 (iterate here)  (one check    (1 shot/hypothesis)    (locked)
                  per hyp.)
```

**Asset class:** US equities (S&P 500 universe)
**Universe:** Top 150 stocks by 30-day average dollar volume, selected from
in-sample data (2010–2018) only — future liquidity never influences universe.
**Development:** 2010–2016 (~1,760 trading days) — iterate freely here.
**IS Holdback:** 2017–2018 (~504 trading days) — checked ONCE per hypothesis,
never during parameter exploration within a hypothesis.
**Walk-forward:** 2019–2021 (3 expanding folds) — one shot per hypothesis.
**Holdout:** 2022–2024 — locked until the single final run.

---

## Agent Snapshots

Two files in `.agent/` serve as revert points. They are written automatically
by the harness — never copy or edit them manually.

| File                     | Written when                            | Use for                                                    |
| ------------------------ | --------------------------------------- | ---------------------------------------------------------- |
| `best_dev_agent.py`      | `--in-sample` sets a new DEV score high | Reverting a bad DEV experiment within a hypothesis         |
| `best_holdback_agent.py` | `--check-holdback` passes the gate      | Reverting after a holdback failure or walk-forward failure |

**The key distinction:**

- `best_dev_agent.py` may contain code whose holdback has never been checked.
  Use it only to undo a bad within-hypothesis tweak.
- `best_holdback_agent.py` is always a fully validated state (DEV + holdback
  gate passed). This is the safe revert point when a hypothesis is closed.

On a fresh run with no prior history, neither file exists yet. The first
`--in-sample` run creates `best_dev_agent.py`. The first passing
`--check-holdback` creates `best_holdback_agent.py`.

---

## Score Formula

```
score = sharpe
      - max(0, (turnover - 0.3) * 0.5)      # penalize high turnover
      - max(0, (|max_drawdown| - 0.20) * 2)  # penalize deep drawdowns
```

---

## Experiment Loop

### Step 1 — Read current state

Read the current editable section of `agent.py`.
Read `results.csv` to understand what has been tried and what failed.
Inspect the description field of every past entry.
Do not repeat an approach unless you have a genuinely new economic
justification. Parameter sweeps with no new rationale are forbidden.

### Step 2 — Declare hypothesis

Before writing any code, write a rationale using this exact format:

```
HYPOTHESIS:  The economic mechanism you are testing (one sentence).
CHANGE:      Exactly what code changes, one sentence.
WHY:         Specific mechanism — cite a paper or name a known market effect.
             "To see if it helps" is not valid.
PARAMETER:   If changing a parameter: what range was considered, and why
             this specific value — not just "I tried 200".
```

If you cannot write a convincing WHY that names a specific mechanism,
do not run the experiment. Move to the next hypothesis instead.

### Step 3 — Iterate on DEV only

Run experiments freely against the DEV period (2010–2016):

```bash
python agent.py --in-sample \
  --desc "HYPOTHESIS: <n>. CHANGE: <what>. WHY: <why>."
```

**The holdback result is suppressed during `--in-sample` runs.**
You will only see DEV metrics: sharpe, drawdown, turnover, score.
Do NOT attempt to infer holdback performance from any output.
Iterate purely on DEV until you are satisfied with the result for this
hypothesis and ready to commit.

Change ONE thing at a time.

A result on DEV counts as an improvement only if ALL hold:

- `score` improves over the best known DEV score
- `excess_sharpe` > 0.15

If improved: keep the change, increment improvement counter.
If not: revert immediately:

```bash
cp .agent/best_dev_agent.py agent.py
python agent.py --in-sample --desc "revert check"
```

Score must match `.agent/best_dev_score.json` before continuing.

### Step 4 — Holdback gate (once per hypothesis, never during iteration)

Only after you are fully satisfied with the DEV result and ready to commit
the hypothesis, run the holdback check:

```bash
python agent.py --check-holdback \
  --desc "HYPOTHESIS: <n>. HOLDBACK CHECK."
```

**This is a one-shot check. You may not adjust parameters and re-check.**
One check, one verdict. Running `--check-holdback` means you are committed.

Holdback gate — ALL must hold:

| Condition                                   | Threshold |
| ------------------------------------------- | --------- |
| `holdback_sharpe >= decay_min × dev_sharpe` | 0.50×     |
| `holdback_excess > excess_min`              | 0.10      |

If gate fails → revert entire hypothesis, permanently closed:

```bash
cp .agent/best_holdback_agent.py agent.py
python agent.py --in-sample --desc "revert after holdback fail"
```

If gate passes → eligible for walk-forward (step 5).

### Step 5 — Walk-forward (one shot per hypothesis)

Before running, declare in your output:

```
HYPOTHESIS:     <n>
MECHANISM:      <one sentence economic justification>
BEST_IS_SCORE:  <score from last_result.json>
COMMITTED:      yes — walk-forward result is final for this hypothesis
```

Then run:

```bash
python agent.py --walk-forward \
  --desc "HYPOTHESIS: <n>. COMMITTED: yes. WF-CHECK: #N"
```

Walk-forward pass criteria — ALL must hold:

| Metric                | Threshold | Notes                            |
| --------------------- | --------- | -------------------------------- |
| `mean_oos_sharpe`     | ≥ 0.80    |                                  |
| `mean_excess_sharpe`  | ≥ 0.15    | Must meaningfully beat benchmark |
| `oos_is_sharpe_ratio` | ≥ 0.50    | OOS Sharpe must be ≥ 50% of dev  |
| `fold_pass_count`     | ≥ 2       | At least 2 of 3 folds clear 0.60 |

**Forbidden after seeing a walk-forward result:**

- Tweaking parameters because WF score was close to threshold
- Re-running WF with "just one small fix"
- Revisiting a failed hypothesis with different params but the same mechanism

After walk-forward:

- PASS → freeze code as best, move to next hypothesis
- FAIL → revert to `best_holdback_agent.py`, hypothesis permanently closed

### Step 6 — Repeat

Repeat until a stopping criterion is met:

- `mean_oos_sharpe >= 1.2` on a passed walk-forward — target achieved
- 100 iterations — time limit

Do not stop for any other reason. Do not ask whether to continue.
Hitting a high DEV score alone is not a stopping condition — only a
passing walk-forward result counts as genuine progress.

---

## Research Directive

**Target metrics (after 10bps transaction costs):**

- Sharpe ratio > 1.2 (annualized)
- Excess Sharpe > 0.3 over equal-weight buy-and-hold of same universe
- Max drawdown < 25%
- Monthly turnover < 50%
- Long-only (no shorting)

**Available data keys in `get_signals()` and `get_position_sizes()`:**

- `data["close"]` — daily adjusted close prices
- `data["high"]` — daily adjusted highs
- `data["low"]` — daily adjusted lows
- `data["volume"]` — daily share volume
- `data["spy"]` — SPY close prices

All keys are pd.DataFrame with shape (dates × tickers) except `spy` which
is (dates × 1). All fields are already loaded and cached.

---

## How to Run

```bash
# Install deps
pip install -e .

# Establish baseline on DEV period only (holdback suppressed)
python agent.py --in-sample --desc "baseline: 12-1 momentum + SPY 200d regime"

# Iterate on DEV — holdback not shown
python agent.py --in-sample \
  --desc "HYPOTHESIS: vol scaling. CHANGE: inverse vol weighting. WHY: dampens exposure before drawdowns."

# Holdback gate — run ONCE per hypothesis when ready to commit
python agent.py --check-holdback \
  --desc "HYPOTHESIS: vol scaling. HOLDBACK CHECK."

# Walk-forward — run once per hypothesis, committed
python agent.py --walk-forward \
  --desc "HYPOTHESIS: vol scaling. COMMITTED: yes. WF-CHECK: #1"

# Final holdout — only once, run by human
python agent.py --holdout --desc "final holdout: best strategy"
```
