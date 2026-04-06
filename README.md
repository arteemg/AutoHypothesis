# quant-autoagent

Single-file autonomous quant strategy search. Faithful to the
[autoagent](https://github.com/kevinrgu/autoagent) pattern — one file,
one boundary, the meta-agent edits above it, the harness lives below it.

## Structure

```
agent.py
  ┌─────────────────────────────────────┐
  │ EDITABLE HARNESS                    │  ← meta-agent rewrites this
  │   get_signals(data)                 │
  │   get_position_sizes(signals, data) │
  └─────────────────────────────────────┘
  ══ FIXED ADAPTER BOUNDARY ═══════════════
  ┌─────────────────────────────────────┐
  │ FIXED HARNESS                       │  ← never touched
  │   UNIVERSE_SIZE = 150               │  ← fixed, do not change
  │   TRAIN_END = "2022-12-31"          │  ← fixed, do not change
  │   HOLDOUT_START = "2023-01-01"      │  ← fixed, do not change
  │   load_data()                       │
  │   simulate()                        │
  │   compute_metrics()                 │
  │   walk_forward()                    │
  │   run_loop()  ← meta-agent loop     │
  │   CLI entry point                   │
  └─────────────────────────────────────┘

program.md     ← YOU edit this
results.tsv    ← auto-generated experiment log
last_result.json ← auto-generated, last backtest output
.agent/best_agent.py ← auto-saved best strategy snapshot
```

## Quick start

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

# Test the baseline strategy
python agent.py --in-sample

# Run walk-forward on baseline
python agent.py --walk-forward

# Launch the meta-agent for 50 iterations (runs overnight)
python agent.py --loop 50

# Final reality check on held-out 2023 data (run once, at the end)
python agent.py --holdout
```

## The only file you edit

`program.md` — change the research directive, the hypothesis space, the score
formula, or the target metrics. The meta-agent reads this to decide what to try.

## Score formula

```
score = sharpe
      - max(0, (turnover - 0.3) * 0.5)
      - max(0, (|max_drawdown| - 0.20) * 2)
```

A score > 1.2 is good. > 1.5 is excellent.

## Stopping criteria (automatic)

| Condition | Meaning |
|---|---|
| score > 1.8 | Target achieved |
| 50 iterations | Time limit |
| 3 consecutive walk-forward failures | Overfit — stop |

## Customizing

- **Different asset class**: swap the tickers in `SP500_TICKERS` for ETFs, futures, crypto
- **Different score formula**: edit `compute_metrics()` in the fixed section AND update `program.md`
- **Different train/test split**: change `TRAIN_END` in the fixed section — note this invalidates cross-experiment comparisons
- **More universe stocks**: change `UNIVERSE_SIZE` in the fixed section — note this invalidates cross-experiment comparisons

> ⚠️ `UNIVERSE_SIZE`, `TRAIN_END`, and `HOLDOUT_START` are hardcoded in the **fixed section** and are intentionally not editable by the meta-agent. Changing them manually will invalidate comparisons across experiments.
