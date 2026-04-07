"""
Single-file quant strategy harness.

The meta-agent edits everything above the FIXED ADAPTER BOUNDARY.
The fixed section below handles data loading, simulation, scoring, and walk-forward.

Run:
  python agent.py --in-sample       # dev (2010-2016) only — holdback suppressed
  python agent.py --check-holdback  # holdback gate — run once per hypothesis
  python agent.py --walk-forward    # expanding walk-forward on 2019-2021
  python agent.py --holdout         # final holdout eval (run once, at the very end)

Design:
  1. Point-in-time universe  — universe selected from IS dollar volume only.
  2. Four-way split          — DEV (2010-2016), HOLDBACK (2017-2018),
                               WF (2019-2021), HOLDOUT (2022+).
  3. Holdback gate           — only triggered by --check-holdback, NOT by
                               --in-sample. Agent iterates on DEV without
                               ever seeing holdback results. One check per
                               hypothesis; one verdict, no re-checks.
  4. True walk-forward       — expanding training window; test folds are
                               strictly inside the WF period.
  5. WF pass criteria        — requires mean_oos_sharpe >= 0.80,
                               mean_excess_sharpe >= 0.15,
                               oos_is_sharpe_ratio >= 0.50,
                               fold_pass_count >= 2.
  6. Benchmark tracking      — every result logs excess_sharpe vs equal-weight
                               buy-and-hold of the same universe.
"""

from __future__ import annotations
import hashlib
import json
import argparse
import importlib
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pickle

import pandas as pd
import numpy as np


# ============================================================================
# EDITABLE SECTION — meta-agent modifies only this section
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
    spy = data["spy"].squeeze()   # DataFrame (N×1) → Series (N,)

    # 12-1 momentum: 12-month return, skip most recent month
    momentum = close.pct_change(252).shift(21)
    ranked = momentum.rank(axis=1, pct=True)
    top_momentum = ranked > 0.90

    # Market regime: only long when SPY is above its 200-day MA
    spy_200d = spy.rolling(200).mean()
    in_bull_market = spy > spy_200d

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

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
# Do not change — altering any of these invalidates cross-experiment comparison.

UNIVERSE_SIZE = 150

# Development: the only period the meta-agent ever optimises against
DEV_START = "2010-01-01"
DEV_END = "2016-12-31"

# IS Holdback: evaluated ONLY via --check-holdback, never during --in-sample
HOLDBACK_START = "2017-01-01"
HOLDBACK_END = "2018-12-31"

# Convenience alias: full in-sample = DEV + HOLDBACK
IS_START = DEV_START
IS_END = HOLDBACK_END

# Walk-forward: test folds drawn exclusively from this window
WF_START = "2019-01-01"
WF_END = "2021-12-31"

# Holdout: never touched until the single final run
HOLDOUT_START = "2022-01-01"

# Holdback gate thresholds (checked once per hypothesis)
HOLDBACK_DECAY_MIN = 0.50   # holdback_sharpe >= this * dev_sharpe
HOLDBACK_EXCESS_MIN = 0.10   # holdback_excess must exceed this

# Walk-forward pass thresholds
WF_MEAN_SHARPE_MIN = 0.80
WF_MEAN_EXCESS_MIN = 0.15
WF_DECAY_RATIO_MIN = 0.50   # mean_oos_sharpe / dev_sharpe
WF_FOLD_SHARPE_MIN = 0.60   # per-fold threshold for fold_pass_count
WF_FOLD_PASS_MIN = 2      # number of folds that must clear WF_FOLD_SHARPE_MIN

TRANSACTION_COST = 0.0010
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
AGENT_DIR = ROOT / ".agent"
RESULTS_CSV = ROOT / "results.csv"
LAST_RESULT = ROOT / "last_result.json"
DATA_DIR.mkdir(exist_ok=True)
AGENT_DIR.mkdir(exist_ok=True)

# ── Gate token helpers ────────────────────────────────────────────────────────
#
# last_passed_holdback.json records the MD5 hash of agent.py at the moment a
# --check-holdback run passed. --walk-forward checks this token before running.
# If the file is absent or the hash doesn't match the current code, WF is
# blocked. This makes it mechanically impossible to run WF on code that has
# not passed holdback.
#
# The hash is of the file bytes, not the mtime, so copying best_dev_agent.py
# back (which updates mtime) still produces the correct hash if the content
# is unchanged.

GATE_TOKEN_FILE = AGENT_DIR / "last_passed_holdback.json"


def _code_hash() -> str:
    """MD5 of the current agent.py file contents."""
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()


def _write_gate_token(dev_sharpe: float, holdback_sharpe: float) -> None:
    """Write gate token after a passing --check-holdback."""
    token = {
        "code_hash":       _code_hash(),
        "timestamp":       datetime.now().isoformat(),
        "dev_sharpe":      dev_sharpe,
        "holdback_sharpe": holdback_sharpe,
    }
    GATE_TOKEN_FILE.write_text(json.dumps(token, indent=2))
    print(f"  ✓ Gate token written  hash={token['code_hash'][:8]}…")


def _invalidate_gate_if_code_changed() -> None:
    """
    Called at the top of every --in-sample run.
    If agent.py has changed since the last passing holdback, delete the token
    so that --walk-forward cannot proceed until --check-holdback is re-run.
    """
    if not GATE_TOKEN_FILE.exists():
        return
    try:
        token = json.loads(GATE_TOKEN_FILE.read_text())
    except Exception:
        GATE_TOKEN_FILE.unlink(missing_ok=True)
        return
    if token.get("code_hash") != _code_hash():
        GATE_TOKEN_FILE.unlink()
        print("  [gate token invalidated — code changed since last holdback pass]")


def _assert_gate_token_valid() -> None:
    """
    Called at the top of every --walk-forward run.
    Exits with a clear error if no valid gate token exists for the current code.
    """
    if not GATE_TOKEN_FILE.exists():
        print("✗ WALK-FORWARD BLOCKED")
        print("  No passing holdback gate found for the current code.")
        print("  Run --check-holdback first and ensure it passes.")
        sys.exit(1)

    try:
        token = json.loads(GATE_TOKEN_FILE.read_text())
    except Exception:
        print("✗ WALK-FORWARD BLOCKED — gate token unreadable.")
        sys.exit(1)

    if token.get("code_hash") != _code_hash():
        print("✗ WALK-FORWARD BLOCKED")
        print("  agent.py has changed since the last passing holdback check.")
        print("  Run --check-holdback again for the current code.")
        sys.exit(1)

    print(f"  ✓ Gate token valid  hash={token['code_hash'][:8]}…  "
          f"holdback_sharpe={token.get('holdback_sharpe', '?')}")


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
    """
    Load raw OHLCV for the full ticker list.
    Universe selection happens later in select_universe() using only IS data.
    """
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

    close = raw["Close"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    volume = raw["Volume"].copy()

    min_obs = int(0.80 * len(close))
    valid = close.columns[close.notna().sum() >= min_obs].tolist()
    if "SPY" not in valid:
        valid.append("SPY")

    close = close[valid].ffill()
    high = high[valid].ffill()
    low = low[valid].ffill()
    volume = volume[valid].ffill()

    data = {
        "close":   close,
        "high":    high,
        "low":     low,
        "volume":  volume,
        "spy":     close[["SPY"]],
        "tickers": [t for t in valid if t != "SPY"],
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Cached {len(valid)} tickers (including SPY).")
    return data


# ── Universe & split helpers ──────────────────────────────────────────────────

def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index <= end)]


def select_universe(raw: dict) -> dict:
    """
    Point-in-time universe selection.

    Universe is chosen by average 30-day dollar volume computed using
    full in-sample data (2010-2018) only. Future liquidity figures never
    influence which stocks appear in the universe.
    """
    is_close = _slice(raw["close"],  IS_START, IS_END)
    is_volume = _slice(raw["volume"], IS_START, IS_END)

    dv = (is_close * is_volume).rolling(30).mean()
    non_spy = [t for t in is_close.columns if t != "SPY"]
    top_n = dv[non_spy].mean().nlargest(UNIVERSE_SIZE).index.tolist()
    universe = top_n + ["SPY"]

    tradeable = [t for t in universe if t != "SPY"]
    return {
        "close":   raw["close"][tradeable],
        "high":    raw["high"][tradeable],
        "low":     raw["low"][tradeable],
        "volume":  raw["volume"][tradeable],
        "spy":     raw["spy"],
        "tickers": tradeable,
    }


def four_way_split(data: dict) -> tuple[dict, dict, dict, dict]:
    """
    Split data into four non-overlapping periods.

    Returns:
        dev      — 2010-01-01 → 2016-12-31  (agent optimises here only)
        holdback — 2017-01-01 → 2018-12-31  (one-shot decay gate per hypothesis)
        wf       — 2019-01-01 → 2021-12-31  (walk-forward, one shot per hypothesis)
        holdout  — 2022-01-01 → ...          (locked until final run)
    """
    def _make(start, end):
        return {
            k: _slice(v, start, end) if isinstance(v, pd.DataFrame) else v
            for k, v in data.items()
        }

    dev = _make(DEV_START,      DEV_END)
    holdback = _make(HOLDBACK_START, HOLDBACK_END)
    wf = _make(WF_START,       WF_END)
    holdout = _make(HOLDOUT_START,  "2099-12-31")
    return dev, holdback, wf, holdout


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_returns(data: dict) -> pd.Series:
    """Equal-weight buy-and-hold of all tradeable tickers in `data`."""
    return data["close"].pct_change().fillna(0.0).mean(axis=1)


def benchmark_sharpe(returns: pd.Series) -> float:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    return round(ann_ret / ann_vol, 4) if ann_vol > 0 else 0.0


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(data: dict, signals_fn, sizes_fn) -> tuple[pd.Series, pd.DataFrame]:
    close = data["close"]
    signals = signals_fn(data)
    weights = sizes_fn(signals, data)

    signals = signals.reindex(close.index).fillna(False)
    weights = weights.reindex(close.index).fillna(0.0)

    weights_exec = weights.shift(1).fillna(0.0)
    stock_returns = close.pct_change().fillna(0.0)
    gross = (weights_exec * stock_returns).sum(axis=1)
    cost = weights_exec.diff().abs().sum(axis=1).fillna(0.0) * TRANSACTION_COST
    return gross - cost, weights_exec


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, weights: pd.DataFrame,
                    bm_sharpe: float = 0.0) -> dict:
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0, "max_drawdown": 0, "annual_return": 0,
                "turnover": 0, "score": 0, "excess_sharpe": 0,
                "benchmark_sharpe": bm_sharpe}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + returns).cumprod()
    max_drawdown = ((cum - cum.cummax()) / cum.cummax()).min()

    monthly_w = weights.resample("ME").last()
    turnover = monthly_w.diff().abs().sum(axis=1).mean()

    score = (sharpe
             - max(0, (turnover - 0.3) * 0.5)
             - max(0, (abs(max_drawdown) - 0.20) * 2))

    return {
        "sharpe":           round(sharpe, 4),
        "max_drawdown":     round(max_drawdown, 4),
        "annual_return":    round(ann_ret, 4),
        "turnover":         round(turnover, 4),
        "score":            round(score, 4),
        "benchmark_sharpe": round(bm_sharpe, 4),
        "excess_sharpe":    round(sharpe - bm_sharpe, 4),
        "n_days":           len(returns),
    }


# ── Holdback gate (one-shot, explicit flag only) ──────────────────────────────

def run_holdback_gate(pit_data: dict, dev_metrics: dict,
                      signals_fn, sizes_fn) -> dict:
    """
    Evaluate the IS holdback gate.

    Called ONLY by --check-holdback. Never called during --in-sample.

    Gate passes only if ALL hold:
      holdback_sharpe >= HOLDBACK_DECAY_MIN * dev_sharpe
      holdback_excess >= HOLDBACK_EXCESS_MIN
    """
    # Use full IS context so rolling windows are valid on holdback dates
    full_is = {
        k: _slice(v, IS_START, IS_END) if isinstance(v, pd.DataFrame) else v
        for k, v in pit_data.items()
    }
    hb_ret_full, hb_wgt_full = simulate(full_is, signals_fn, sizes_fn)
    hb_ret = hb_ret_full[hb_ret_full.index >= HOLDBACK_START]
    hb_wgt = hb_wgt_full[hb_wgt_full.index >= HOLDBACK_START]
    hb_bm_full = benchmark_returns(full_is)
    hb_bm = benchmark_sharpe(hb_bm_full[hb_bm_full.index >= HOLDBACK_START])

    hb_m = compute_metrics(hb_ret, hb_wgt, bm_sharpe=hb_bm)

    dev_sharpe = dev_metrics.get("sharpe", 0)
    hb_sharpe = hb_m.get("sharpe", 0)
    hb_excess = hb_m.get("excess_sharpe", 0)

    decay_ok = hb_sharpe >= HOLDBACK_DECAY_MIN * dev_sharpe
    excess_ok = hb_excess >= HOLDBACK_EXCESS_MIN
    gate_pass = decay_ok and excess_ok

    reasons = []
    if not decay_ok:
        reasons.append(
            f"holdback_sharpe {hb_sharpe:.3f} < "
            f"{HOLDBACK_DECAY_MIN} * dev_sharpe {dev_sharpe:.3f} "
            f"= {HOLDBACK_DECAY_MIN * dev_sharpe:.3f}"
        )
    if not excess_ok:
        reasons.append(
            f"holdback_excess {hb_excess:.3f} < {HOLDBACK_EXCESS_MIN}"
        )

    if gate_pass:
        print(f"  ✓ Holdback gate PASSED  "
              f"hb_sharpe={hb_sharpe:.3f}  "
              f"hb_excess={hb_excess:+.3f}")
    else:
        print(f"  ✗ Holdback gate FAILED: {'; '.join(reasons)}")
        print(
            "  → Revert to best_holdback_agent.py. This hypothesis is permanently closed.")

    return {
        "holdback_gate_pass":   gate_pass,
        "dev_sharpe":           round(dev_sharpe, 4),
        "holdback_sharpe":      round(hb_sharpe, 4),
        "holdback_excess":      round(hb_excess, 4),
        "holdback_decay_ratio": round(hb_sharpe / dev_sharpe
                                      if dev_sharpe != 0 else 0, 4),
        "holdback":             hb_m,
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(dev: dict, wf: dict, signals_fn, sizes_fn,
                 n_folds: int = 3) -> dict:
    """
    Expanding walk-forward over the WF period (2019-2021).

    For n_folds=3 over ~756 WF trading days (~252 days per fold):
      Fold 1 — context: DEV only           -> tests ~2019
      Fold 2 — context: DEV + fold-1 WF    -> tests ~2020
      Fold 3 — context: DEV + folds 1-2    -> tests ~2021

    Test folds are evaluated in isolation — OOS returns are only those
    from the unseen test window. Training context provides rolling-window
    history without leaking future WF data into signal computation.
    """
    wf_dates = wf["close"].index
    n_wf_days = len(wf_dates)
    fold_days = n_wf_days // n_folds
    folds = []

    for i in range(n_folds):
        test_start_idx = i * fold_days
        test_end_idx = (i + 1) * fold_days if i < n_folds - 1 else n_wf_days
        test_dates = wf_dates[test_start_idx:test_end_idx]
        wf_seen_dates = wf_dates[:test_start_idx]

        def _build(key):
            parts = [dev[key]]
            if len(wf_seen_dates) > 0:
                parts.append(wf[key][wf[key].index.isin(wf_seen_dates)])
            return pd.concat(parts).sort_index()

        def _full(key):
            return pd.concat([
                _build(key),
                wf[key][wf[key].index.isin(test_dates)],
            ]).sort_index()

        full_ctx = {
            "close":   _full("close"),
            "high":    _full("high"),
            "low":     _full("low"),
            "volume":  _full("volume"),
            "spy":     _full("spy"),
            "tickers": dev["tickers"],
        }

        try:
            ret_full, wgt_full = simulate(full_ctx, signals_fn, sizes_fn)

            ret_test = ret_full[ret_full.index.isin(test_dates)]
            wgt_test = wgt_full[wgt_full.index.isin(test_dates)]

            bm_full = benchmark_returns(full_ctx)
            bm_test = bm_full[bm_full.index.isin(test_dates)]
            bm_shr = benchmark_sharpe(bm_test)

            m = compute_metrics(ret_test, wgt_test, bm_sharpe=bm_shr)
            m["fold"] = i + 1
            m["train_days"] = len(dev["close"]) + len(wf_seen_dates)
            m["test_days"] = len(test_dates)
            m["period"] = f"{test_dates[0].date()} -> {test_dates[-1].date()}"
            m["train_end"] = (str(wf_seen_dates[-1].date())
                              if len(wf_seen_dates) else
                              str(dev["close"].index[-1].date()))

        except Exception as e:
            m = {"fold": i + 1, "error": str(e), "sharpe": 0,
                 "score": 0, "excess_sharpe": 0}

        folds.append(m)
        print(f"  Fold {i+1}/{n_folds}: {m.get('period', '')}  "
              f"sharpe={m.get('sharpe', 0):.3f}  "
              f"excess={m.get('excess_sharpe', 0):+.3f}")

    sharpes = [f["sharpe"] for f in folds if "sharpe" in f]
    excess_sharpes = [f["excess_sharpe"]
                      for f in folds if "excess_sharpe" in f]

    mean_oos = round(np.mean(sharpes),        4) if sharpes else 0
    mean_exc = round(np.mean(excess_sharpes), 4) if excess_sharpes else 0
    std_oos = round(np.std(sharpes),         4) if sharpes else 0

    dev_sharpe_ref = 0.0
    best_score_path = AGENT_DIR / "best_score.json"
    if best_score_path.exists():
        try:
            dev_sharpe_ref = json.loads(
                best_score_path.read_text()).get("dev_sharpe", 0.0)
        except Exception:
            pass
    oos_is_ratio = round(
        mean_oos / dev_sharpe_ref if dev_sharpe_ref > 0 else 0.0, 4)

    fold_pass_count = sum(1 for s in sharpes if s >= WF_FOLD_SHARPE_MIN)

    passed = bool(
        mean_oos >= WF_MEAN_SHARPE_MIN and
        mean_exc >= WF_MEAN_EXCESS_MIN and
        oos_is_ratio >= WF_DECAY_RATIO_MIN and
        fold_pass_count >= WF_FOLD_PASS_MIN
    )

    reasons_failed = []
    if mean_oos < WF_MEAN_SHARPE_MIN:
        reasons_failed.append(
            f"mean_oos_sharpe {mean_oos} < {WF_MEAN_SHARPE_MIN}")
    if mean_exc < WF_MEAN_EXCESS_MIN:
        reasons_failed.append(
            f"mean_excess_sharpe {mean_exc} < {WF_MEAN_EXCESS_MIN}")
    if oos_is_ratio < WF_DECAY_RATIO_MIN:
        reasons_failed.append(
            f"oos_is_sharpe_ratio {oos_is_ratio} < {WF_DECAY_RATIO_MIN}")
    if fold_pass_count < WF_FOLD_PASS_MIN:
        reasons_failed.append(
            f"fold_pass_count {fold_pass_count} < {WF_FOLD_PASS_MIN}")

    if passed:
        print(f"  ✓ Walk-forward PASSED  mean_oos={mean_oos:.3f}  "
              f"excess={mean_exc:+.3f}  decay_ratio={oos_is_ratio:.2f}  "
              f"folds_passed={fold_pass_count}/3")
    else:
        print(f"  ✗ Walk-forward FAILED: {'; '.join(reasons_failed)}")

    return {
        "type":                "walk_forward",
        "folds":               folds,
        "mean_oos_sharpe":     mean_oos,
        "std_oos_sharpe":      std_oos,
        "mean_excess_sharpe":  mean_exc,
        "oos_is_sharpe_ratio": oos_is_ratio,
        "fold_pass_count":     fold_pass_count,
        "pass":                passed,
        "fail_reasons":        reasons_failed,
    }


# ── Results logging ───────────────────────────────────────────────────────────

def _log_result(result: dict, mode: str, description: str = "") -> None:
    import csv

    if mode == "walk_forward":
        metrics = result.get("dev", {})
        oos_note = (
            f"[OOS sharpe={result.get('mean_oos_sharpe', '')} "
            f"excess={result.get('mean_excess_sharpe', '')} "
            f"decay_ratio={result.get('oos_is_sharpe_ratio', '')} "
            f"folds_passed={result.get('fold_pass_count', '')} "
            f"pass={result.get('pass', '')}]"
        )
        description = f"{oos_note} {description}".strip()
    else:
        metrics = result

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "mode",
                "sharpe", "max_drawdown", "annual_return", "turnover", "score",
                "benchmark_sharpe", "excess_sharpe",
                "dev_sharpe", "holdback_sharpe", "holdback_excess",
                "holdback_gate_pass",
                "description",
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode,
            metrics.get("sharpe",           ""),
            metrics.get("max_drawdown",      ""),
            metrics.get("annual_return",     ""),
            metrics.get("turnover",          ""),
            metrics.get("score",             "") or "",
            metrics.get("benchmark_sharpe",  ""),
            metrics.get("excess_sharpe",     ""),
            result.get("dev_sharpe",         ""),
            result.get("holdback_sharpe",    ""),
            result.get("holdback_excess",    ""),
            result.get("holdback_gate_pass", ""),
            description,
        ])


# ── Best-agent snapshots ──────────────────────────────────────────────────────
#
# Two separate snapshot files with distinct purposes:
#
#   best_dev_agent.py      — highest DEV score seen so far, regardless of
#                            holdback status. Updated after every --in-sample
#                            run that sets a new DEV score high. Used to revert
#                            within a hypothesis after a bad DEV experiment.
#
#   best_holdback_agent.py — highest DEV score among runs whose holdback gate
#                            has explicitly passed. Updated only after a
#                            successful --check-holdback. Used to revert after
#                            a holdback failure or a walk-forward failure —
#                            always points to a fully validated state.

BEST_DEV_AGENT_FILE = AGENT_DIR / "best_dev_agent.py"
BEST_HOLDBACK_AGENT_FILE = AGENT_DIR / "best_holdback_agent.py"


def _save_best_if_new(mode: str, result: dict) -> None:
    """
    Maintain two independent agent snapshots after every scored run.

    best_dev_agent.py      — updated on --in-sample when DEV score improves.
    best_holdback_agent.py — updated on --check-holdback when gate passes
                             AND the DEV score of this run beats all previous
                             holdback-validated runs.
    """
    import shutil

    if mode not in ("in_sample", "check_holdback"):
        return
    if not RESULTS_CSV.exists():
        return

    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return

    def _float(val) -> float:
        return round(float(pd.to_numeric(val, errors="coerce")), 4)

    # ── 1. best_dev_agent.py — best DEV score, no holdback requirement ────────
    if mode == "in_sample":
        dev_rows = df[df["mode"] == "in_sample"].copy()
        dev_rows["score"] = pd.to_numeric(
            dev_rows["score"], errors="coerce").fillna(0)

        if not dev_rows.empty:
            best_idx = dev_rows["score"].idxmax()
            latest_idx = dev_rows.index[-1]

            if latest_idx == best_idx:
                shutil.copy(Path(__file__), BEST_DEV_AGENT_FILE)
                best_row = dev_rows.loc[best_idx]
                summary = {
                    "score":         _float(best_row["score"]),
                    "sharpe":        _float(best_row.get("sharpe", 0)),
                    "excess_sharpe": _float(best_row.get("excess_sharpe", 0)),
                    "dev_sharpe":    _float(best_row.get("sharpe", 0)),
                    "description":   str(best_row.get("description", "")),
                    "timestamp":     str(best_row.get("timestamp", "")),
                }
                (AGENT_DIR / "best_dev_score.json").write_text(
                    json.dumps(summary, indent=2))
                print(f"  ✓ New best_dev_agent.py  "
                      f"score={summary['score']:.4f}  "
                      f"excess={summary['excess_sharpe']:+.4f}")

    # ── 2. best_holdback_agent.py — best holdback-validated DEV score ─────────
    if mode == "check_holdback" and result.get("holdback_gate_pass", False):
        hb_rows = df[df["mode"] == "check_holdback"].copy()
        hb_rows["score"] = pd.to_numeric(
            hb_rows["score"], errors="coerce").fillna(0)

        # Only consider rows where the gate passed
        hb_rows["holdback_gate_pass"] = (
            hb_rows["holdback_gate_pass"].astype(str).str.lower())
        passed_rows = hb_rows[hb_rows["holdback_gate_pass"] == "true"]

        if not passed_rows.empty:
            best_idx = passed_rows["score"].idxmax()
            latest_idx = passed_rows.index[-1]

            if latest_idx == best_idx:
                shutil.copy(Path(__file__), BEST_HOLDBACK_AGENT_FILE)
                best_row = passed_rows.loc[best_idx]
                summary = {
                    "score":            _float(best_row["score"]),
                    "sharpe":           _float(best_row.get("sharpe", 0)),
                    "excess_sharpe":    _float(best_row.get("excess_sharpe", 0)),
                    "dev_sharpe":       _float(best_row.get("sharpe", 0)),
                    "holdback_sharpe":  _float(best_row.get("holdback_sharpe", 0)),
                    "holdback_excess":  _float(best_row.get("holdback_excess", 0)),
                    "holdback_gate_pass": True,
                    "description":      str(best_row.get("description", "")),
                    "timestamp":        str(best_row.get("timestamp", "")),
                }
                (AGENT_DIR / "best_holdback_score.json").write_text(
                    json.dumps(summary, indent=2))
                # Also update best_score.json — used by walk-forward decay ratio
                (AGENT_DIR / "best_score.json").write_text(
                    json.dumps(summary, indent=2))
                print(f"  ✓ New best_holdback_agent.py  "
                      f"score={summary['score']:.4f}  "
                      f"holdback_sharpe={summary['holdback_sharpe']:.4f}  "
                      f"excess={summary['excess_sharpe']:+.4f}")

        # Write gate token regardless of whether this is the all-time best —
        # any passing holdback unlocks walk-forward for the current code.
        _write_gate_token(
            dev_sharpe=result.get("dev_sharpe", result.get("sharpe", 0)),
            holdback_sharpe=result.get("holdback_sharpe", 0),
        )


# ── Single-run entry point ────────────────────────────────────────────────────

def run_once(mode: str = "in_sample", description: str = "") -> dict:
    mod_name = Path(__file__).stem
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)

    signals_fn = mod.get_signals
    sizes_fn = mod.get_position_sizes

    raw_data = load_data()
    pit_data = select_universe(raw_data)
    dev, holdback, wf, holdout = four_way_split(pit_data)

    if mode == "holdout":
        print("Running holdout eval (2022-2024)...")
        returns, weights = simulate(holdout, signals_fn, sizes_fn)
        bm_shr = benchmark_sharpe(benchmark_returns(holdout))
        result = compute_metrics(returns, weights, bm_sharpe=bm_shr)
        result["type"] = "holdout"

        for label, start, end in [
            ("bear_2022",      "2022-01-01", "2022-12-31"),
            ("bull_2023_2024", "2023-01-01", "2024-12-31"),
        ]:
            sub = {
                k: _slice(v, start, end) if isinstance(v, pd.DataFrame) else v
                for k, v in holdout.items()
            }
            r_sub, w_sub = simulate(sub, signals_fn, sizes_fn)
            bm_sub = benchmark_sharpe(benchmark_returns(sub))
            result[label] = compute_metrics(r_sub, w_sub, bm_sharpe=bm_sub)

    elif mode == "walk_forward":
        # Gate check — exits with clear error if holdback not passed for current code
        _assert_gate_token_valid()
        print("Running walk-forward (2019-2021, 3 folds)...")
        result = walk_forward(dev, wf, signals_fn, sizes_fn)

        dev_ret, dev_wgt = simulate(dev, signals_fn, sizes_fn)
        bm_shr = benchmark_sharpe(benchmark_returns(dev))
        result["dev"] = compute_metrics(dev_ret, dev_wgt, bm_sharpe=bm_shr)

    elif mode == "check_holdback":
        # One-shot holdback gate — run only when agent is ready to commit
        print("Running IS holdback gate check (2017-2018)...")
        print("WARNING: This is a one-shot check. No re-checks allowed.")

        dev_ret, dev_wgt = simulate(dev, signals_fn, sizes_fn)
        dev_bm = benchmark_sharpe(benchmark_returns(dev))
        dev_m = compute_metrics(dev_ret, dev_wgt, bm_sharpe=dev_bm)

        gate = run_holdback_gate(pit_data, dev_m, signals_fn, sizes_fn)

        result = {**dev_m, **gate}
        result["type"] = "check_holdback"

        print(f"  dev  sharpe={dev_m['sharpe']:.3f}  "
              f"excess={dev_m['excess_sharpe']:+.3f}  "
              f"score={dev_m['score']:.3f}")

    else:
        # --in-sample: DEV period only. Holdback is NOT evaluated.
        # Invalidate gate token if code has changed since last passing holdback.
        _invalidate_gate_if_code_changed()
        print("Running dev backtest (2010-2016)...")
        print("  [Holdback suppressed — use --check-holdback when ready to commit]")

        dev_ret, dev_wgt = simulate(dev, signals_fn, sizes_fn)
        dev_bm = benchmark_sharpe(benchmark_returns(dev))
        dev_m = compute_metrics(dev_ret, dev_wgt, bm_sharpe=dev_bm)

        result = dev_m
        result["type"] = "in_sample"
        result["holdback_gate_pass"] = None   # not checked yet
        result["holdback_sharpe"] = None
        result["holdback_excess"] = None

        print(f"  dev  sharpe={dev_m['sharpe']:.3f}  "
              f"excess={dev_m['excess_sharpe']:+.3f}  "
              f"score={dev_m['score']:.3f}")

    LAST_RESULT.write_text(json.dumps(result, indent=2, default=str))
    _log_result(result, mode, description)
    _save_best_if_new(mode, result)
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward",    action="store_true")
    parser.add_argument("--holdout",         action="store_true")
    parser.add_argument("--in-sample",       action="store_true")
    parser.add_argument("--check-holdback",  action="store_true",
                        help="One-shot holdback gate check — run once per hypothesis only")
    parser.add_argument("--desc", default="",
                        help="Short description logged to results.csv")
    args = parser.parse_args()

    if args.holdout:
        print(json.dumps(run_once("holdout",        args.desc), indent=2, default=str))
    elif args.walk_forward:
        print(json.dumps(run_once("walk_forward",   args.desc), indent=2, default=str))
    elif args.check_holdback:
        print(json.dumps(run_once("check_holdback", args.desc), indent=2, default=str))
    else:
        print(json.dumps(run_once("in_sample",      args.desc), indent=2, default=str))
