"""
Single-file quant strategy harness.

The meta-agent edits everything above the FIXED ADAPTER BOUNDARY.
The fixed section below handles data loading, simulation, scoring, and the run loop.

Run:
  python agent.py                  # single in-sample backtest
  python agent.py --walk-forward   # 5-fold walk-forward validation
  python agent.py --holdout        # final holdout eval (use sparingly)
  python agent.py --loop 50        # meta-agent loop for N iterations
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ============================================================================
# EDITABLE HARNESS — meta-agent modifies this section
# ============================================================================

def get_signals(data: dict) -> pd.DataFrame:
    """
    Generate entry signals.

    Args:
        data: dict with keys close, high, low, volume, spy
              all pd.DataFrame with shape (dates, tickers)

    Returns:
        pd.DataFrame[bool] — True means long this stock today
    """
    close = data["close"]
    spy   = data["spy"].squeeze()        # DataFrame (N×1) → Series (N,)

    # 12-1 momentum: 12-month return, skip most recent month
    momentum = close.pct_change(252).shift(21)

    # Market regime: only long when SPY is above its 200-day MA
    spy_200d = spy.rolling(200).mean()
    in_bull_market = (spy > spy_200d)    # boolean Series

    # Take top 10% by momentum score
    ranked = momentum.rank(axis=1, pct=True)
    top_momentum = ranked > 0.90

    signals = top_momentum & in_bull_market.values.reshape(-1, 1)
    return signals.fillna(False)


def get_position_sizes(signals: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Convert signals to portfolio weights.

    Args:
        signals: boolean DataFrame from get_signals()
        data:    same data dict

    Returns:
        pd.DataFrame[float] — portfolio weights, rows must sum to <= 1.0
    """
    # Equal weight among selected stocks
    n_selected = signals.sum(axis=1).replace(0, np.nan)
    weights = signals.astype(float).div(n_selected, axis=0).fillna(0.0)
    return weights


# ============================================================================
# FIXED ADAPTER BOUNDARY — do not modify below this line
# ============================================================================

from datetime import timedelta
import pickle
from pathlib import Path
from datetime import datetime
import warnings
import sys
import subprocess
import shutil
import re
import os
import json
import importlib
import argparse

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

UNIVERSE_SIZE = 150            # fixed — do not change, invalidates cross-experiment comparison
TRAIN_END     = "2022-12-31"   # shrinks in-sample, gives more holdout
HOLDOUT_START = "2023-01-01"   # ~2+ years of holdout through today

TRANSACTION_COST = 0.0010
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
AGENT_DIR     = ROOT / ".agent"
RESULTS_TSV   = ROOT / "results.tsv"
LAST_RESULT   = ROOT / "last_result.json"
BEST_SNAPSHOT = AGENT_DIR / "best_agent.py"
DATA_DIR.mkdir(exist_ok=True)
AGENT_DIR.mkdir(exist_ok=True)

# ── Tickers ───────────────────────────────────────────────────────────────────

SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "LLY",
    "JPM", "V", "XOM", "AVGO", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO", "PEP",
    "COST", "TMO", "WMT", "MCD", "CSCO", "ABT", "CRM", "BAC", "ACN", "LIN", "ADBE",
    "NFLX", "DHR", "TXN", "NEE", "PM", "CMCSA", "VZ", "RTX", "HON", "AMGN", "ORCL",
    "IBM", "QCOM", "T", "UPS", "LOW", "INTU", "SPGI", "GS", "CAT", "ELV", "BLK", "AXP",
    "SYK", "GILD", "MDT", "DE", "ADI", "REGN", "VRTX", "MMM", "C", "MS", "ISRG", "ADP",
    "PLD", "CI", "BSX", "PANW", "KLAC", "LRCX", "MU", "AMAT", "SNPS", "CDNS", "MELI",
    "ZTS", "SO", "DUK", "AON", "PNC", "USB", "ICE", "F", "GM", "FDX", "NSC", "UNP",
    "CSX", "WM", "ECL", "EMR", "ITW", "ETN", "APD", "SHW", "MCO", "CTAS", "ROK", "CMI",
    "PH", "GD", "NOC", "LMT", "BA", "HUM", "CNC", "CVS", "WBA", "MCK", "ABC", "CAH",
    "JNJ", "PFE", "BMY", "BIIB", "ILMN", "IDXX", "A", "DXCM", "EW", "HCA", "DGX",
    "LH", "IQV", "RMD", "STE", "HOLX", "TDY", "BAX", "BDX", "ZBH", "TECH", "HSIC",
    "COO", "PODD", "SPY",
]

# ── Data loading ──────────────────────────────────────────────────────────────

CACHE_FILE = DATA_DIR / "ohlcv_cache.pkl"


def _cache_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
    return age < timedelta(days=1)


def load_data(start: str = "2009-01-01", end: str = "2025-12-31",
              force: bool = False) -> dict:
    if not force and _cache_fresh():
        print("Loading from cache...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"Fetching data ({start} → {end})...")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    tickers = list(dict.fromkeys(SP500_TICKERS))
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=True, threads=True)

    close  = raw["Close"].copy()
    high   = raw["High"].copy()
    low    = raw["Low"].copy()
    volume = raw["Volume"].copy()

    min_obs = int(0.80 * len(close))
    valid   = close.columns[close.notna().sum() >= min_obs].tolist()
    if "SPY" not in valid:
        valid.append("SPY")

    close  = close[valid].ffill()
    high   = high[valid].ffill()
    low    = low[valid].ffill()
    volume = volume[valid].ffill()

    dv      = (close * volume).rolling(30).mean()
    non_spy = [t for t in valid if t != "SPY"]
    top_n   = dv[non_spy].mean().nlargest(UNIVERSE_SIZE).index.tolist()
    universe = top_n + ["SPY"]

    close  = close[universe]
    high   = high[universe]
    low    = low[universe]
    volume = volume[universe]

    tradeable = [t for t in universe if t != "SPY"]
    data = {
        "close":   close[tradeable],
        "high":    high[tradeable],
        "low":     low[tradeable],
        "volume":  volume[tradeable],
        "spy":     close[["SPY"]],
        "tickers": tradeable,
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Cached {len(tradeable)} tickers.")
    return data


def split_data(data: dict, train_end: str) -> tuple[dict, dict]:
    def _before(df): return df[df.index <= train_end]
    def _after(df):  return df[df.index > train_end]
    train   = {k: _before(v) if isinstance(v, pd.DataFrame) else v for k, v in data.items()}
    holdout = {k: _after(v)  if isinstance(v, pd.DataFrame) else v for k, v in data.items()}
    return train, holdout


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(data: dict, signals_fn, sizes_fn) -> tuple[pd.Series, pd.DataFrame]:
    close   = data["close"]
    signals = signals_fn(data)
    weights = sizes_fn(signals, data)

    signals = signals.reindex(close.index).fillna(False)
    weights = weights.reindex(close.index).fillna(0.0)

    weights_exec  = weights.shift(1).fillna(0.0)
    stock_returns = close.pct_change().fillna(0.0)
    gross         = (weights_exec * stock_returns).sum(axis=1)
    cost          = weights_exec.diff().abs().sum(axis=1).fillna(0.0) * TRANSACTION_COST
    return gross - cost, weights_exec


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, weights: pd.DataFrame) -> dict:
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0, "max_drawdown": 0, "annual_return": 0,
                "turnover": 0, "score": 0}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    cum          = (1 + returns).cumprod()
    max_drawdown = ((cum - cum.cummax()) / cum.cummax()).min()

    monthly_w = weights.resample("ME").last()
    turnover  = monthly_w.diff().abs().sum(axis=1).mean()

    score = (sharpe
             - max(0, (turnover - 0.3) * 0.5)
             - max(0, (abs(max_drawdown) - 0.20) * 2))

    return {
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_drawdown, 4),
        "annual_return": round(ann_ret, 4),
        "turnover":      round(turnover, 4),
        "score":         round(score, 4),
        "n_days":        len(returns),
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(data: dict, signals_fn, sizes_fn, n_folds: int = 5) -> dict:
    dates     = data["close"].index
    fold_days = len(dates) // (n_folds + 1)
    folds     = []

    for i in range(n_folds):
        test_start = fold_days * (i + 1)
        test_end   = min(test_start + fold_days, len(dates) - 1)
        test_dates = dates[test_start:test_end]

        slice_data = {k: v[v.index.isin(test_dates)] if isinstance(v, pd.DataFrame) else v
                      for k, v in data.items()}
        try:
            returns, weights = simulate(slice_data, signals_fn, sizes_fn)
            m = compute_metrics(returns, weights)
            m["fold"]   = i + 1
            m["period"] = f"{test_dates[0].date()} → {test_dates[-1].date()}"
        except Exception as e:
            m = {"fold": i + 1, "error": str(e), "sharpe": 0, "score": 0}
        folds.append(m)

    sharpes = [f["sharpe"] for f in folds if "sharpe" in f]
    return {
        "type":            "walk_forward",
        "folds":           folds,
        "mean_oos_sharpe": round(np.mean(sharpes), 4) if sharpes else 0,
        "std_oos_sharpe":  round(np.std(sharpes), 4)  if sharpes else 0,
        "pass":            bool(np.mean(sharpes) >= 0.8) if sharpes else False,
    }


# ── Single-run entry point ────────────────────────────────────────────────────

def run_once(mode: str = "in_sample") -> dict:
    mod_name = Path(__file__).stem
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)

    signals_fn = mod.get_signals
    sizes_fn   = mod.get_position_sizes

    all_data        = load_data()
    train, holdout  = split_data(all_data, TRAIN_END)

    if mode == "holdout":
        print("⚠️  HOLDOUT eval")
        returns, weights = simulate(holdout, signals_fn, sizes_fn)
        result = compute_metrics(returns, weights)
        result["type"] = "holdout"

    elif mode == "walk_forward":
        print("Running walk-forward validation...")
        result = walk_forward(train, signals_fn, sizes_fn)
        returns, weights = simulate(train, signals_fn, sizes_fn)
        result["in_sample"] = compute_metrics(returns, weights)

    else:
        print("Running in-sample backtest...")
        returns, weights = simulate(train, signals_fn, sizes_fn)
        result = compute_metrics(returns, weights)
        result["type"] = "in_sample"

    LAST_RESULT.write_text(json.dumps(result, indent=2))
    return result


# ── Meta-agent loop ───────────────────────────────────────────────────────────

def _init_tsv():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "iter\ttimestamp\tdescription\tsharpe\tmax_drawdown\tturnover\tscore\tkept\n"
        )


def _log(iter_num, description, metrics, kept):
    with open(RESULTS_TSV, "a") as f:
        f.write(
            f"{iter_num}\t"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\t"
            f"{description[:800].replace(chr(9), ' ')}\t"
            f"{metrics.get('sharpe', 0):.4f}\t"
            f"{metrics.get('max_drawdown', 0):.4f}\t"
            f"{metrics.get('turnover', 0):.4f}\t"
            f"{metrics.get('score', 0):.4f}\t"
            f"{'YES' if kept else 'NO'}\n"
        )


def _call_claude(messages: list, system: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp   = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def _extract_python(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "def get_signals" in text and "def get_position_sizes" in text:
        return text.strip()
    return None


def _rewrite_editable_section(new_code: str) -> bool:
    this_file = Path(__file__)
    original  = this_file.read_text()
    boundary  = "# ============================================================================\n# FIXED ADAPTER BOUNDARY"

    if boundary not in original:
        print("ERROR: boundary marker not found in agent.py")
        return False

    fixed_section = original[original.index(boundary):]

    header = '''\
"""
Single-file quant strategy harness.

The meta-agent edits everything above the FIXED ADAPTER BOUNDARY.
The fixed section below handles data loading, simulation, scoring, and the run loop.

Run:
  python agent.py                  # single in-sample backtest
  python agent.py --walk-forward   # 5-fold walk-forward validation
  python agent.py --holdout        # final holdout eval (use sparingly)
  python agent.py --loop 50        # meta-agent loop for N iterations
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ============================================================================
# EDITABLE HARNESS — meta-agent modifies this section
# ============================================================================

'''
    # Strip __future__ imports — already in header
    clean = "\n".join(
        l for l in new_code.splitlines()
        if not l.strip().startswith("from __future__")
        and not l.strip().startswith("import pandas")
        and not l.strip().startswith("import numpy")
    )

    new_file = header + clean.strip() + "\n\n\n" + fixed_section
    this_file.write_text(new_file)
    return True


def run_loop(max_iters: int = 50):
    _init_tsv()

    program = (ROOT / "program.md").read_text()

    system_prompt = f"""You are a quantitative research agent. Your job is to iteratively
improve a trading strategy by rewriting the EDITABLE HARNESS section of agent.py.

The editable section must contain exactly two functions:
- get_signals(data) -> pd.DataFrame[bool]
- get_position_sizes(signals, data) -> pd.DataFrame[float]

STRICT RULES:
1. Output COMPLETE replacement code inside ```python ... ```
2. Start with the # EDITABLE HARNESS comment, then the two functions — nothing else
3. Do NOT include UNIVERSE_SIZE, TRAIN_END, or HOLDOUT_START — those are fixed in the harness
4. Do NOT add import statements — pandas and numpy are already imported as pd and np
5. data dict keys: close, high, low, volume, spy — all pd.DataFrame (dates x tickers)
6. No lookahead bias: never use .shift(-N) with negative N
7. Weights must sum to <= 1.0 per row
8. Before the code block, write short reasoning in exactly this format (2 lines, no more):
   CHANGE: one sentence on what you are changing
   WHY: one sentence on why it should improve the score

RESEARCH DIRECTIVE:
{program}
"""

    print("=" * 60)
    print("QUANT AUTOAGENT")
    print("=" * 60)
    print("\nBaseline backtest...")
    best_metrics = run_once("in_sample")
    best_score   = best_metrics.get("score", -999)
    shutil.copy(Path(__file__), BEST_SNAPSHOT)
    _log(0, "baseline: 12-1 momentum + SPY 200d regime", best_metrics, kept=True)
    print(f"Baseline score: {best_score:.4f}")

    conversation  = []
    improvements  = 0
    wf_failures   = 0

    for iteration in range(1, max_iters + 1):
        print(f"\n{'='*60}\nITERATION {iteration}/{max_iters}\n{'='*60}")

        this_file_text = Path(__file__).read_text()
        boundary_marker = "# ============================================================================\n# FIXED ADAPTER BOUNDARY"
        editable_section = this_file_text[:this_file_text.index(boundary_marker)].strip()

        results_recent = "\n".join(RESULTS_TSV.read_text().strip().split("\n")[-11:]) \
                         if RESULTS_TSV.exists() else ""

        user_msg = f"""## Current state
Best score: {best_score:.4f}
Best metrics: {json.dumps(best_metrics, indent=2)}
Successful improvements: {improvements}
Consecutive walk-forward failures: {wf_failures}

## Current editable section
```python
{editable_section}
```

## Recent experiment log
```
{results_recent}
```

Propose and implement your next improvement.
Output reasoning first, then the COMPLETE replacement editable section in a ```python block.
"""
        conversation.append({"role": "user", "content": user_msg})

        print("Querying Claude...")
        response = _call_claude(conversation, system_prompt)
        conversation.append({"role": "assistant", "content": response})

        reasoning = re.split(r"```python", response)[0].strip()
        print(f"\nReasoning: {reasoning[:400]}{'...' if len(reasoning) > 400 else ''}")

        new_code = _extract_python(response)
        if not new_code:
            print("⚠️  No valid Python found. Skipping.")
            continue
        if "def get_signals" not in new_code or "def get_position_sizes" not in new_code:
            print("⚠️  Missing required functions. Skipping.")
            continue

        if not _rewrite_editable_section(new_code):
            continue

        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--in-sample"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )

        if not LAST_RESULT.exists() or proc.returncode != 0:
            print(f"❌ Backtest failed:\n{proc.stderr[-500:]}")
            shutil.copy(BEST_SNAPSHOT, Path(__file__))
            _log(iteration, "REVERTED (error)", {}, kept=False)
            continue

        metrics = json.loads(LAST_RESULT.read_text())
        score   = metrics.get("score", -999)
        desc    = reasoning[:800].replace("\n", " ")

        print(f"Score: {score:.4f} | Sharpe: {metrics.get('sharpe', 0):.3f} | "
              f"DD: {metrics.get('max_drawdown', 0):.1%} | TO: {metrics.get('turnover', 0):.2f}")

        if score > best_score:
            print(f"✅ {best_score:.4f} → {score:.4f}")
            best_score   = score
            best_metrics = metrics
            shutil.copy(Path(__file__), BEST_SNAPSHOT)
            _log(iteration, desc, metrics, kept=True)
            improvements += 1

            if improvements % 5 == 0:
                print("\n🔍 Walk-forward check...")
                subprocess.run(
                    [sys.executable, str(Path(__file__)), "--walk-forward"],
                    cwd=ROOT, capture_output=True, text=True, timeout=600,
                )
                if LAST_RESULT.exists():
                    wf       = json.loads(LAST_RESULT.read_text())
                    passed   = wf.get("pass", False)
                    mean_oos = wf.get("mean_oos_sharpe", 0)
                    print(f"Walk-forward: {'✅ PASS' if passed else '❌ FAIL'} "
                          f"(mean OOS Sharpe: {mean_oos:.3f})")
                    wf_failures = 0 if passed else wf_failures + 1
        else:
            print(f"❌ No improvement. Reverting.")
            shutil.copy(BEST_SNAPSHOT, Path(__file__))
            _log(iteration, desc, metrics, kept=False)

        if best_score >= 1.8:
            print(f"\n🎉 Target reached (score={best_score:.4f}). Done.")
            break
        if wf_failures >= 3:
            print(f"\n⚠️  3 walk-forward failures — likely overfit. Stopping.")
            break

    print(f"\n{'='*60}\nFINAL BEST SCORE: {best_score:.4f}")
    print(json.dumps(best_metrics, indent=2))
    print(f"\nBest strategy: {BEST_SNAPSHOT}")
    print(f"Experiment log: {RESULTS_TSV}")
    print(f"\nHoldout eval: cp {BEST_SNAPSHOT} {Path(__file__)} && python agent.py --holdout")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--holdout",      action="store_true")
    parser.add_argument("--in-sample",    action="store_true")
    parser.add_argument("--loop", type=int, default=None,
                        help="Run meta-agent loop for N iterations")
    args = parser.parse_args()

    if args.loop is not None:
        run_loop(max_iters=args.loop)
    elif args.holdout:
        print(json.dumps(run_once("holdout"), indent=2))
    elif args.walk_forward:
        print(json.dumps(run_once("walk_forward"), indent=2))
    else:
        print(json.dumps(run_once("in_sample"), indent=2))
