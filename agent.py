"""
Single-file quant strategy harness.

The meta-agent edits everything above the FIXED ADAPTER BOUNDARY.
The fixed section below handles data loading, simulation, scoring, and walk-forward.

Run:
  python agent.py                  # single in-sample backtest
  python agent.py --walk-forward   # expanding walk-forward on 2019-2021
  python agent.py --holdout        # final holdout eval (run once, at the very end)
  
  1. Point-in-time universe  — universe selected from IS dollar volume only.
  2. Explicit three-way split — IS (2010-2018), WF (2019-2021), holdout (2022+).
  3. True walk-forward       — expanding training window; test folds are
     strictly inside the WF period, never touching IS or holdout.
  4. Benchmark tracking      — every result logs excess_sharpe vs equal-weight
     buy-and-hold of the same universe over the same period.
"""

from __future__ import annotations
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
    spy = data["spy"].squeeze()   # DataFrame (N×1) → Series (N,)

    # 12-1 momentum: 12-month return, skip most recent month
    momentum = close.pct_change(252).shift(21)

    # Market regime: only long when SPY is above its 200-day MA
    spy_200d = spy.rolling(200).mean()
    in_bull_market = spy > spy_200d

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

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
# Do not change — altering any of these invalidates cross-experiment comparison.

UNIVERSE_SIZE = 150

# In-sample: the only period the meta-agent ever optimises against
IS_START = "2010-01-01"
IS_END = "2018-12-31"

# Walk-forward: test folds are drawn exclusively from this window
WF_START = "2019-01-01"
WF_END = "2021-12-31"

# Kept for legacy CLI compatibility — equals WF_END
TRAIN_END = WF_END

# Holdout: never touched until the single final run
HOLDOUT_START = "2022-01-01"

TRANSACTION_COST = 0.0010
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
AGENT_DIR = ROOT / ".agent"
RESULTS_CSV = ROOT / "results.csv"
LAST_RESULT = ROOT / "last_result.json"
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
    in-sample data only (2010–2018). Future liquidity figures never
    influence which stocks appear in the training universe.

    The full time series for the selected tickers is returned — the
    selection criterion used only IS data, but all periods remain
    available for later slicing into IS / WF / holdout.
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


def three_way_split(data: dict) -> tuple[dict, dict, dict]:
    """
    FIX 2 — Explicit three-way split.

    Returns:
        insample — 2010-01-01 → 2018-12-31  (meta-agent optimises here only)
        wf       — 2019-01-01 → 2021-12-31  (walk-forward test territory)
        holdout  — 2022-01-01 → ...          (locked until the very final run)
    """
    def _make(start, end):
        return {
            k: _slice(v, start, end) if isinstance(v, pd.DataFrame) else v
            for k, v in data.items()
        }

    insample = _make(IS_START,      IS_END)
    wf = _make(WF_START,      WF_END)
    holdout = _make(HOLDOUT_START, "2099-12-31")
    return insample, wf, holdout


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


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(insample: dict, wf: dict, signals_fn, sizes_fn,
                 n_folds: int = 3) -> dict:
    """

    Test folds are drawn exclusively from the wf dict (2021–2022).
    Training for each fold starts at IS_START and expands through all
    WF data seen before that fold's test window begins.

    For n_folds=3 over ~504 WF trading days (~168 days per fold):
      Fold 1 — context: IS only          → tests Jan–Jun 2021
      Fold 2 — context: IS + fold-1 WF   → tests Jun–Nov 2021
      Fold 3 — context: IS + folds 1-2   → tests Nov 2021–Dec 2022

    Why context includes IS history:
      get_signals() uses pct_change(252) and rolling(252). The first WF
      date (Jan 2021) needs lookbacks reaching into 2019. Concatenating
      IS + prior WF rows provides the full rolling history without leaking
      future WF data into the signal.
    """
    wf_dates = wf["close"].index
    n_wf_days = len(wf_dates)
    fold_days = n_wf_days // n_folds
    folds = []

    for i in range(n_folds):
        test_start_idx = i * fold_days
        test_end_idx = (i + 1) * fold_days if i < n_folds - 1 else n_wf_days
        test_dates = wf_dates[test_start_idx:test_end_idx]
        # WF rows seen before this fold
        wf_seen_dates = wf_dates[:test_start_idx]

        # Build context: IS history + any WF rows before the test window
        def _build(key):
            parts = [insample[key]]
            if len(wf_seen_dates) > 0:
                parts.append(wf[key][wf[key].index.isin(wf_seen_dates)])
            return pd.concat(parts).sort_index()

        # Extend context through the test window so signals can be computed
        # on test dates (rolling windows need the preceding rows)
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
            "tickers": insample["tickers"],
        }

        try:
            ret_full, wgt_full = simulate(full_ctx, signals_fn, sizes_fn)

            # Evaluate on test dates only — this is the genuine OOS result
            ret_test = ret_full[ret_full.index.isin(test_dates)]
            wgt_test = wgt_full[wgt_full.index.isin(test_dates)]

            bm_full = benchmark_returns(full_ctx)
            bm_test = bm_full[bm_full.index.isin(test_dates)]
            bm_shr = benchmark_sharpe(bm_test)

            m = compute_metrics(ret_test, wgt_test, bm_sharpe=bm_shr)
            m["fold"] = i + 1
            m["train_days"] = len(insample["close"]) + len(wf_seen_dates)
            m["test_days"] = len(test_dates)
            m["period"] = f"{test_dates[0].date()} → {test_dates[-1].date()}"
            m["train_end"] = (str(wf_seen_dates[-1].date())
                              if len(wf_seen_dates) else
                              str(insample["close"].index[-1].date()))

        except Exception as e:
            m = {"fold": i + 1, "error": str(e), "sharpe": 0,
                 "score": 0, "excess_sharpe": 0}

        folds.append(m)
        print(f"  Fold {i + 1}/{n_folds}: {m.get('period', '')}  "
              f"sharpe={m.get('sharpe', 0):.3f}  "
              f"excess={m.get('excess_sharpe', 0):+.3f}")

    sharpes = [f["sharpe"] for f in folds if "sharpe" in f]
    excess_sharpes = [f["excess_sharpe"]
                      for f in folds if "excess_sharpe" in f]

    return {
        "type":               "walk_forward",
        "folds":              folds,
        "mean_oos_sharpe":    round(np.mean(sharpes),        4) if sharpes else 0,
        "std_oos_sharpe":     round(np.std(sharpes),         4) if sharpes else 0,
        "mean_excess_sharpe": round(np.mean(excess_sharpes), 4) if excess_sharpes else 0,
        # pass = Sharpe threshold AND beating the equal-weight benchmark
        "pass": bool(
            np.mean(sharpes) >= 0.8 and np.mean(excess_sharpes) > 0.0
        ) if sharpes else False,
    }


# ── Results logging ───────────────────────────────────────────────────────────

def _log_result(result: dict, mode: str, description: str = "") -> None:
    import csv

    if mode == "walk_forward":
        metrics = result.get("in_sample", {})
        oos_note = (
            f"[OOS sharpe={result.get('mean_oos_sharpe', '')} "
            f"excess={result.get('mean_excess_sharpe', '')} "
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
                "description",
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode,
            metrics.get("sharpe", ""),
            metrics.get("max_drawdown", ""),
            metrics.get("annual_return", ""),
            metrics.get("turnover", ""),
            metrics.get("score", "") or "",
            metrics.get("benchmark_sharpe", ""),
            metrics.get("excess_sharpe", ""),
            description,
        ])


# ── Best-agent snapshot ───────────────────────────────────────────────────────

BEST_AGENT_FILE = AGENT_DIR / "best_agent.py"


def _save_best_if_new(mode: str) -> None:
    """
    After each run, read results.csv and check whether the most recent
    in-sample row is the all-time best by score. If so, snapshot the
    current agent.py and write a summary to .agent/best_agent.py.

    CSV is the single source of truth — no separate state file needed.
    Walk-forward and holdout rows are ignored; only in_sample rows count.
    """
    if mode != "in_sample":
        return
    if not RESULTS_CSV.exists():
        return

    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return

    is_rows = df[df["mode"] == "in_sample"].copy()
    if is_rows.empty:
        return

    is_rows["score"] = pd.to_numeric(
        is_rows["score"], errors="coerce").fillna(0)
    is_rows["excess_sharpe"] = pd.to_numeric(
        is_rows["excess_sharpe"], errors="coerce").fillna(0)

    best_idx = is_rows["score"].idxmax()
    best_row = is_rows.loc[best_idx]
    latest_row = is_rows.iloc[-1]

    # Current run is the best if it matches the row with the highest score
    if latest_row.name != best_idx:
        return

    import shutil
    shutil.copy(Path(__file__), BEST_AGENT_FILE)

    summary = {
        "score":         round(float(best_row["score"]), 4),
        "excess_sharpe": round(float(best_row["excess_sharpe"]), 4),
        "sharpe":        round(float(pd.to_numeric(best_row.get("sharpe", 0), errors="coerce")), 4),
        "description":   str(best_row.get("description", "")),
        "timestamp":     str(best_row.get("timestamp", "")),
    }

    summary_path = AGENT_DIR / "best_score.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"  ✓ New best → .agent/best_agent.py  "
          f"score={summary['score']:.4f}  "
          f"excess={summary['excess_sharpe']:+.4f}")


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
    pit_data = select_universe(raw_data)        # FIX 1
    insample, wf, holdout = three_way_split(pit_data)        # FIX 2

    if mode == "holdout":
        print("⚠️  HOLDOUT eval — do not iterate after this.")
        returns, weights = simulate(holdout, signals_fn, sizes_fn)
        bm_shr = benchmark_sharpe(benchmark_returns(holdout))
        result = compute_metrics(returns, weights, bm_sharpe=bm_shr)
        result["type"] = "holdout"

    elif mode == "walk_forward":
        print("Running walk-forward (2019–2021, 3 folds)...")
        result = walk_forward(insample, wf, signals_fn, sizes_fn)  # FIX 3
        returns, weights = simulate(insample, signals_fn, sizes_fn)
        bm_shr = benchmark_sharpe(benchmark_returns(insample))
        result["in_sample"] = compute_metrics(
            returns, weights, bm_sharpe=bm_shr)

    else:
        print("Running in-sample backtest (2010–2018)...")
        returns, weights = simulate(insample, signals_fn, sizes_fn)
        bm_shr = benchmark_sharpe(benchmark_returns(insample))
        result = compute_metrics(returns, weights, bm_sharpe=bm_shr)
        result["type"] = "in_sample"

    LAST_RESULT.write_text(json.dumps(result, indent=2))
    _log_result(result, mode, description)
    _save_best_if_new(mode)
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--holdout",      action="store_true")
    parser.add_argument("--in-sample",    action="store_true")
    parser.add_argument("--desc", default="",
                        help="Short description logged to results.csv")
    args = parser.parse_args()

    if args.holdout:
        print(json.dumps(run_once("holdout", args.desc), indent=2))
    elif args.walk_forward:
        print(json.dumps(run_once("walk_forward", args.desc), indent=2))
    else:
        print(json.dumps(run_once("in_sample", args.desc), indent=2))
