"""
BaseIndicators.py

Compute the rolling market indicators used throughout the project for
all five indices in one run: DAX, WIG, OMXT, OMXR and OMXV.

Algorithms:
- logarithmic close-to-close returns;
- 30-day rolling annualized standard deviation;
- rolling realized variance and realized volatility;
- rolling autocorrelation at lags 1, 5 and 10;
- rolling unbiased skewness;
- rolling unbiased excess kurtosis;
- rolling 30-day return z-scores;
- historical 5% Value-at-Risk.

The script reads raw index files from data/raw/ and writes each index's
complete indicator dataset to base_indicators/{INDEX}_indicators/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "base_indicators"

INDICES = {
    "OMXT": "omxt.txt",
    "OMXR": "omxr.txt",
    "OMXV": "omxv.txt",
    "WIG": "WIG.txt",
    "DAX": "dax.txt",
}

WINDOW = 30
AUTOCORRELATION_LAGS = (1, 5, 10)
TRADING_DAYS = 252


# ============================================================
# DATA LOADING
# ============================================================

def load_market_data(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"max": "high", "min": "low"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.sort_values("date")
        .dropna(subset=["date", "close"])
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    return df


def calculate_log_returns(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


# ============================================================
# ROLLING INDICATORS
# ============================================================

def rolling_volatility(returns, window):
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def rolling_autocorrelation(returns, window, lag):
    def autocorrelation(values):
        x = np.asarray(values, dtype=float)
        if len(x) <= lag:
            return np.nan
        x1, x2 = x[lag:], x[:-lag]
        if np.std(x1) == 0 or np.std(x2) == 0:
            return np.nan
        return np.corrcoef(x1, x2)[0, 1]

    return returns.rolling(window).apply(autocorrelation, raw=True)


def rolling_skewness(returns, window):
    return returns.rolling(window).apply(
        lambda x: skew(x, bias=False),
        raw=True
    )


def rolling_kurtosis(returns, window):
    return returns.rolling(window).apply(
        lambda x: kurtosis(x, fisher=True, bias=False),
        raw=True
    )


def rolling_realized_variance(returns, window):
    return returns.pow(2).rolling(window).sum()


def rolling_realized_volatility(returns, window):
    return np.sqrt(rolling_realized_variance(returns, window))


def rolling_var_5pct(returns, window, loss_convention=False):
    var = returns.rolling(window).quantile(0.05)
    return -var if loss_convention else var


def rolling_z_scores(returns, window):
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (returns - mean) / std


# ============================================================
# CALCULATE ALL INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()
    returns = df["log_return"]

    volatility = rolling_volatility(returns, WINDOW)

    # Keep both names for compatibility with existing downstream scripts.
    df[f"volatility_{WINDOW}d"] = volatility
    df[f"volatility_{WINDOW}d_annualized"] = volatility

    realized_variance = rolling_realized_variance(returns, WINDOW)
    realized_volatility = rolling_realized_volatility(returns, WINDOW)

    df[f"realized_variance_{WINDOW}d"] = realized_variance
    df[f"realized_volatility_{WINDOW}d"] = realized_volatility
    df[f"realized_volatility_{WINDOW}d_annualized"] = (
        realized_volatility * np.sqrt(TRADING_DAYS / WINDOW)
    )

    for lag in AUTOCORRELATION_LAGS:
        df[f"autocorrelation_lag{lag}_{WINDOW}d"] = (
            rolling_autocorrelation(returns, WINDOW, lag)
        )

    df[f"skewness_{WINDOW}d"] = rolling_skewness(returns, WINDOW)
    df[f"kurtosis_excess_{WINDOW}d"] = rolling_kurtosis(returns, WINDOW)
    df[f"z_score_{WINDOW}d"] = rolling_z_scores(returns, WINDOW)

    df[f"VaR_5pct_{WINDOW}d"] = rolling_var_5pct(
        returns, WINDOW, loss_convention=False
    )
    df[f"VaR_5pct_loss_{WINDOW}d"] = rolling_var_5pct(
        returns, WINDOW, loss_convention=True
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df, index_name):
    output_dir = OUTPUT_ROOT / f"{index_name}_indicators"
    output_dir.mkdir(parents=True, exist_ok=True)

    complete_file = output_dir / "index_with_indicators.txt"
    df.to_csv(complete_file, index=False)

    z_scores = df[["date", "log_return", f"z_score_{WINDOW}d"]]
    z_scores.to_csv(
        output_dir / "index_z_scores.txt",
        index=False
    )

    return complete_file


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for index_name, filename in INDICES.items():
        input_path = RAW_DATA_DIR / filename

        print("=" * 70)
        print(f"Processing {index_name}")
        print(f"Input: {input_path}")

        if not input_path.exists():
            print(f"WARNING: file not found; skipping {index_name}")
            continue

        df = load_market_data(input_path)
        df = calculate_log_returns(df)
        df = calculate_indicators(df)

        output_path = save_results(df, index_name)

        print(f"Observations: {len(df)}")
        print(
            f"Period: {df['date'].min().date()} "
            f"to {df['date'].max().date()}"
        )
        print(f"Saved: {output_path}")

    print("\nAll available indices processed.")


if __name__ == "__main__":
    main()
