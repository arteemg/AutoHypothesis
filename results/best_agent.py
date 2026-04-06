"""
Single-file quant strategy harness.

The meta-agent edits everything above the FIXED ADAPTER BOUNDARY.
The fixed section below handles data loading, simulation, scoring, and walk-forward.

Run:
  python agent.py                  # single in-sample backtest
  python agent.py --walk-forward   # 5-fold walk-forward validation
  python agent.py --holdout        # final holdout eval (Do not use! Only for final sanity check run by a human.)
"""

from __future__ import annotations
import json
import argparse
import importlib
import sys
import warnings
from datetime import datetime
from pathlib import Path
import pickle
from datetime import timedelta

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
    spy = data["spy"].squeeze()        # DataFrame (N×1) → Series (N,)

    # 12-1 momentum: 12-month return, skip most recent month
    raw_momentum = close.pct_change(252).shift(21)

    # Risk-adjust: divide by trailing 12-month annualized vol (Sharpe-like score)
    # Selects smooth-trending stocks over volatile spike-and-reverters
    vol_12m = close.pct_change().rolling(252).std().shift(21) * np.sqrt(252)
    momentum = raw_momentum / vol_12m.clip(lower=0.05)

    # Hysteresis selection: enter above 90th pct, exit below 80th pct
    # Reduces monthly rotation by keeping "borderline" stocks instead of churning
    ranked = momentum.rank(axis=1, pct=True)
    ranked_monthly = ranked.resample("M").last()  # month-end snapshot

    selection = pd.DataFrame(
        False, index=ranked_monthly.index, columns=close.columns)
    prev = ranked_monthly.iloc[0] > 0.90
    selection.iloc[0] = prev
    for i in range(1, len(ranked_monthly)):
        curr_rank = ranked_monthly.iloc[i]
        stay = prev & (curr_rank > 0.65)  # hold if still above 65th pct
        enter = curr_rank > 0.95           # enter if above 95th pct (top 5%)
        curr = stay | enter
        selection.iloc[i] = curr
        prev = curr

    top_momentum = selection.reindex(close.index, method="ffill").fillna(False)

    # No secondary filter

    return top_momentum.fillna(False)


def get_position_sizes(signals: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Convert signals to portfolio weights.

    Args:
        signals: boolean DataFrame from get_signals()
        data:    same data dict

    Returns:
        pd.DataFrame[float] — portfolio weights, rows must sum to <= 1.0
    """
    spy = data["spy"].squeeze()

    # Equal weight among selected stocks
    n_selected = signals.sum(axis=1).replace(0, np.nan)
    weights = signals.astype(float).div(n_selected, axis=0).fillna(0.0)

    # Layer 1: SPY 200d MA regime — full / 25% exposure
    spy_200d = spy.rolling(200).mean()
    regime_scale = np.where(spy >= spy_200d, 1.0, 0.25)
    regime_scale = pd.Series(regime_scale, index=spy.index).fillna(1.0)

    # Layer 2: vol-normalized scale — reduce further when market vol spikes
    spy_ret = spy.pct_change()
    short_vol = spy_ret.rolling(63).std() * np.sqrt(252)
    long_vol = spy_ret.rolling(252).std() * np.sqrt(252)
    vol_ratio = (long_vol / short_vol.clip(lower=0.01)
                 ).clip(upper=1.0, lower=0.15)
    vol_scale = vol_ratio.resample(
        "M").last().reindex(spy.index, method="ffill")

    weights = weights.mul(regime_scale * vol_scale, axis=0)
    return weights


# ============================================================================
# FIXED ADAPTER BOUNDARY — do not modify below this line
# ============================================================================


warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

# fixed — do not change, invalidates cross-experiment comparison
UNIVERSE_SIZE = 150
TRAIN_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"

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

    dv = (close * volume).rolling(30).mean()
    non_spy = [t for t in valid if t != "SPY"]
    top_n = dv[non_spy].mean().nlargest(UNIVERSE_SIZE).index.tolist()
    universe = top_n + ["SPY"]

    close = close[universe]
    high = high[universe]
    low = low[universe]
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
    def _after(df): return df[df.index > train_end]
    train = {k: _before(v) if isinstance(v, pd.DataFrame)
             else v for k, v in data.items()}
    holdout = {k: _after(v) if isinstance(v, pd.DataFrame)
               else v for k, v in data.items()}
    return train, holdout


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

def compute_metrics(returns: pd.Series, weights: pd.DataFrame) -> dict:
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0, "max_drawdown": 0, "annual_return": 0,
                "turnover": 0, "score": 0}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + returns).cumprod()
    max_drawdown = ((cum - cum.cummax()) / cum.cummax()).min()

    monthly_w = weights.resample("M").last()
    turnover = monthly_w.diff().abs().sum(axis=1).mean()

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
    dates = data["close"].index
    fold_days = len(dates) // (n_folds + 1)
    folds = []

    for i in range(n_folds):
        test_start = fold_days * (i + 1)
        test_end = min(test_start + fold_days, len(dates) - 1)
        test_dates = dates[test_start:test_end]

        slice_data = {k: v[v.index.isin(test_dates)] if isinstance(v, pd.DataFrame) else v
                      for k, v in data.items()}
        try:
            returns, weights = simulate(slice_data, signals_fn, sizes_fn)
            m = compute_metrics(returns, weights)
            m["fold"] = i + 1
            m["period"] = f"{test_dates[0].date()} → {test_dates[-1].date()}"
        except Exception as e:
            m = {"fold": i + 1, "error": str(e), "sharpe": 0, "score": 0}
        folds.append(m)

    sharpes = [f["sharpe"] for f in folds if "sharpe" in f]
    return {
        "type":            "walk_forward",
        "folds":           folds,
        "mean_oos_sharpe": round(np.mean(sharpes), 4) if sharpes else 0,
        "std_oos_sharpe":  round(np.std(sharpes), 4) if sharpes else 0,
        "pass":            bool(np.mean(sharpes) >= 0.8) if sharpes else False,
    }


# ── Results logging ───────────────────────────────────────────────────────────

def _log_result(result: dict, mode: str, description: str = "") -> None:
    import csv
    # For walk-forward, metrics live under result["in_sample"]; append OOS summary to desc
    if mode == "walk_forward":
        metrics = result.get("in_sample", {})
        oos_note = (f"[OOS sharpe={result.get('mean_oos_sharpe', '')} "
                    f"pass={result.get('pass', '')}]")
        description = f"{oos_note} {description}".strip()
    else:
        metrics = result

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "mode", "sharpe", "max_drawdown",
                "annual_return", "turnover", "score", "description",
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode,
            metrics.get("sharpe", ""),
            metrics.get("max_drawdown", ""),
            metrics.get("annual_return", ""),
            metrics.get("turnover", ""),
            metrics.get("score", ""),
            description,
        ])


# ── Single-run entry point ────────────────────────────────────────────────────

def run_once(mode: str = "in_sample", description: str = "") -> dict:
    mod_name = Path(__file__).stem
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)

    signals_fn = mod.get_signals
    sizes_fn = mod.get_position_sizes

    all_data = load_data()
    train, holdout = split_data(all_data, TRAIN_END)

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
    _log_result(result, mode, description)
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--holdout",      action="store_true")
    parser.add_argument("--in-sample",    action="store_true")
    parser.add_argument("--desc",         default="",
                        help="Short description logged to results.csv")
    args = parser.parse_args()

    if args.holdout:
        print(json.dumps(run_once("holdout", args.desc), indent=2))
    elif args.walk_forward:
        print(json.dumps(run_once("walk_forward", args.desc), indent=2))
    else:
        print(json.dumps(run_once("in_sample", args.desc), indent=2))
