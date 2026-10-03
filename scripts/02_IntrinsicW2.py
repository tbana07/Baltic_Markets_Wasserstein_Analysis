"""
IntrinsicW2.py

Compute intrinsic rolling Wasserstein distance for all five indices.

Algorithms:
- logarithmic daily returns;
- consecutive 30-day empirical return distributions;
- exact one-dimensional empirical W_2 using sorted quantile samples;
- a second W_2 after independently standardizing both windows,
  isolating distribution-shape changes from location/scale changes;
- annualized 30-day volatility;
- stress regime defined as observations above the 90th percentile
  of the index-specific volatility distribution.

Outputs preserve the established intrinsic_wasserstein_results structure,
but are now repository-relative:
intrinsic_wasserstein/data/{INDEX}_intrinsic_w2.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_ROOT = PROJECT_ROOT / "intrinsic_wasserstein"
DATA_DIR = OUTPUT_ROOT / "data"
PLOT_DIR = OUTPUT_ROOT / "plots"

START_DATE = "2012-01-02"
END_DATE = "2025-12-30"
WINDOW = 30
STRESS_QUANTILE = 0.90
TRADING_DAYS = 252

INDICES = {
    "OMXT": "omxt.txt",
    "OMXR": "omxr.txt",
    "OMXV": "omxv.txt",
    "WIG": "WIG.txt",
    "DAX": "dax.txt",
}


# ============================================================
# DATA
# ============================================================

def load_market_data(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"max": "high", "min": "low"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.sort_values("date")
        .dropna(subset=["date", "close"])
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


# ============================================================
# WASSERSTEIN FUNCTIONS
# ============================================================

def wasserstein_w2(x, y):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))

    if len(x) != len(y):
        raise ValueError("Empirical W2 requires equal sample sizes.")

    return float(np.sqrt(np.mean((x - y) ** 2)))


def normalize_window(x):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    std = np.std(x, ddof=0)

    if std == 0:
        return np.zeros_like(x)

    return (x - mean) / std


# ============================================================
# INTRINSIC W2
# ============================================================

def calculate_intrinsic_wasserstein(df):
    returns = df["return"].to_numpy()
    dates = df["date"].to_numpy()

    result_dates = []
    raw_w2 = []
    normalized_w2 = []
    volatility = []

    # Two consecutive windows are required.
    for i in range(2 * WINDOW - 1, len(returns)):
        previous_window = returns[
            i - 2 * WINDOW + 1:i - WINDOW + 1
        ]
        current_window = returns[
            i - WINDOW + 1:i + 1
        ]

        raw = wasserstein_w2(previous_window, current_window)

        previous_z = normalize_window(previous_window)
        current_z = normalize_window(current_window)
        normalized = wasserstein_w2(previous_z, current_z)

        current_volatility = (
            np.std(current_window, ddof=1) * np.sqrt(TRADING_DAYS)
        )

        result_dates.append(dates[i])
        raw_w2.append(raw)
        normalized_w2.append(normalized)
        volatility.append(current_volatility)

    result = pd.DataFrame({
        "date": pd.to_datetime(result_dates),
        "w2_raw": raw_w2,
        "w2_normalized": normalized_w2,
        "volatility": volatility,
    })

    threshold = result["volatility"].quantile(STRESS_QUANTILE)

    result["stress_threshold"] = threshold
    result["stress"] = result["volatility"] >= threshold
    result["regime"] = np.where(
        result["stress"], "stress", "normal"
    )
    result["w2_raw_normalized_ratio"] = (
        result["w2_raw"] /
        result["w2_normalized"].replace(0, np.nan)
    )

    return result.set_index("date")


# ============================================================
# SUMMARY
# ============================================================

def create_summary(all_results):
    rows = []

    for market, result in all_results.items():
        normal = result[result["stress"] == False]
        stress = result[result["stress"] == True]

        row = {
            "market": market,
            "total_observations": len(result),
            "normal_observations": len(normal),
            "stress_observations": len(stress),
            "volatility_stress_threshold":
                result["stress_threshold"].iloc[0],
            "mean_volatility_normal":
                normal["volatility"].mean(),
            "mean_volatility_stress":
                stress["volatility"].mean(),
            "expected_w2_raw_normal":
                normal["w2_raw"].mean(),
            "expected_w2_raw_stress":
                stress["w2_raw"].mean(),
            "expected_w2_normalized_normal":
                normal["w2_normalized"].mean(),
            "expected_w2_normalized_stress":
                stress["w2_normalized"].mean(),
            "expected_ratio_normal":
                normal["w2_raw_normalized_ratio"].mean(),
            "expected_ratio_stress":
                stress["w2_raw_normalized_ratio"].mean(),
        }

        row["stress_to_normal_raw_w2"] = (
            row["expected_w2_raw_stress"] /
            row["expected_w2_raw_normal"]
        )
        row["stress_to_normal_normalized_w2"] = (
            row["expected_w2_normalized_stress"] /
            row["expected_w2_normalized_normal"]
        )

        rows.append(row)

    return pd.DataFrame(rows).set_index("market")


def save_summary(summary):
    summary_file = OUTPUT_ROOT / "stress_summary.txt"
    table_file = OUTPUT_ROOT / "intrinsic_w2_stress_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            "INTRINSIC WASSERSTEIN DISTANCE — "
            "VOLATILITY REGIME ANALYSIS\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write(f"Period: {START_DATE} to {END_DATE}\n")
        f.write(f"Rolling window: {WINDOW} trading days\n")
        f.write("Returns: logarithmic returns\n")
        f.write("Volatility: annualized rolling standard deviation\n")
        f.write(
            f"Stress definition: top "
            f"{(1 - STRESS_QUANTILE) * 100:.0f}% "
            "of volatility observations\n\n"
        )
        f.write(
            "Intrinsic W2 compares consecutive 30-day empirical "
            "return distributions.\n"
        )
        f.write(
            "Normalized W2 standardizes each distribution separately "
            "before comparison.\n"
        )

        for market, row in summary.iterrows():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{market}\n")
            f.write("=" * 80 + "\n\n")

            f.write("OBSERVATIONS\n")
            f.write(f"Total:  {int(row['total_observations'])}\n")
            f.write(f"Normal: {int(row['normal_observations'])}\n")
            f.write(f"Stress: {int(row['stress_observations'])}\n\n")

            f.write("VOLATILITY\n")
            f.write(
                f"Stress threshold: {row['volatility_stress_threshold']:.6f}\n"
            )
            f.write(
                f"Mean normal volatility: "
                f"{row['mean_volatility_normal']:.6f}\n"
            )
            f.write(
                f"Mean stress volatility: "
                f"{row['mean_volatility_stress']:.6f}\n\n"
            )

            f.write("RAW W2\n")
            f.write(
                f"E[W2 | normal]: {row['expected_w2_raw_normal']:.10f}\n"
            )
            f.write(
                f"E[W2 | stress]: {row['expected_w2_raw_stress']:.10f}\n"
            )
            f.write(
                f"Stress / normal: "
                f"{row['stress_to_normal_raw_w2']:.6f}\n\n"
            )

            f.write("NORMALIZED W2\n")
            f.write(
                f"E[W2 | normal]: "
                f"{row['expected_w2_normalized_normal']:.10f}\n"
            )
            f.write(
                f"E[W2 | stress]: "
                f"{row['expected_w2_normalized_stress']:.10f}\n"
            )
            f.write(
                f"Stress / normal: "
                f"{row['stress_to_normal_normalized_w2']:.6f}\n\n"
            )

            f.write("RAW / NORMALIZED W2 RATIO\n")
            f.write(
                f"E[ratio | normal]: "
                f"{row['expected_ratio_normal']:.6f}\n"
            )
            f.write(
                f"E[ratio | stress]: "
                f"{row['expected_ratio_stress']:.6f}\n"
            )

    summary.to_csv(
        table_file,
        sep="\t",
        float_format="%.10f"
    )

    return summary_file, table_file


# ============================================================
# PLOTS
# ============================================================

def year_ticks(ax, index):
    years = pd.date_range(
        start=f"{index.min().year}-01-01",
        end=f"{index.max().year}-01-01",
        freq="YS"
    )
    ax.set_xticks(years)
    ax.set_xticklabels(
        [str(y.year) for y in years],
        rotation=45
    )


def save_plots(market, result):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Raw and normalized W2
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(result.index, result["w2_raw"], label="Raw W₂")
    ax.plot(
        result.index,
        result["w2_normalized"],
        label="Normalized W₂"
    )
    ax.set_title(f"{market} — Intrinsic 30-Day Wasserstein Distance")
    ax.set_xlabel("Year")
    ax.set_ylabel("W₂")
    ax.legend()
    ax.grid(True, alpha=0.3)
    year_ticks(ax, result.index)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{market}_intrinsic_w2.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Ratio
    ratio = result["w2_raw_normalized_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )
    finite = ratio.dropna()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        result.index,
        ratio,
        label="Raw W₂ / Normalized W₂"
    )
    if len(finite):
        ymin, ymax = finite.min(), finite.max()
        margin = max((ymax - ymin) * 0.05, abs(ymin) * 0.05, 0.001)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_title(f"{market} — Intrinsic W₂ Raw / Normalized Ratio")
    ax.set_xlabel("Year")
    ax.set_ylabel("Raw W₂ / Normalized W₂")
    ax.legend()
    ax.grid(True, alpha=0.3)
    year_ticks(ax, result.index)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{market}_w2_raw_normalized_ratio.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Volatility
    threshold = result["stress_threshold"].iloc[0]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        result.index,
        result["volatility"],
        label="30-day annualized volatility"
    )
    ax.axhline(
        threshold,
        linestyle="--",
        label="90th percentile (stress threshold)"
    )
    ax.set_title(f"{market} — 30-Day Rolling Volatility and Stress Threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    year_ticks(ax, result.index)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{market}_volatility_stress.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # W2 with stress periods
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(result.index, result["w2_raw"], label="Raw W₂")
    ax.plot(
        result.index,
        result["w2_normalized"],
        label="Normalized W₂"
    )

    stress = result["stress"].to_numpy()
    start = None

    for i, is_stress in enumerate(stress):
        if is_stress and start is None:
            start = result.index[i]
        elif not is_stress and start is not None:
            ax.axvspan(start, result.index[i], alpha=0.15)
            start = None

    if start is not None:
        ax.axvspan(start, result.index[-1], alpha=0.15)

    ax.set_title(f"{market} — Intrinsic W₂ with Volatility Stress Periods")
    ax.set_xlabel("Year")
    ax.set_ylabel("W₂")
    ax.legend()
    ax.grid(True, alpha=0.3)
    year_ticks(ax, result.index)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{market}_w2_stress_periods.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for market, filename in INDICES.items():
        input_path = RAW_DATA_DIR / filename

        print("=" * 70)
        print(f"Processing {market}")
        print(f"Input: {input_path}")

        if not input_path.exists():
            print("WARNING: raw file not found; skipping.")
            continue

        df = load_market_data(input_path)
        df = df[
            (df["date"] >= START_DATE) &
            (df["date"] <= END_DATE)
        ].copy()

        df["return"] = np.log(
            df["close"] / df["close"].shift(1)
        )
        df = df.dropna(subset=["return"])

        result = calculate_intrinsic_wasserstein(df)
        all_results[market] = result

        result.to_csv(
            DATA_DIR / f"{market}_intrinsic_w2.txt",
            sep="\t",
            float_format="%.10f"
        )

        save_plots(market, result)

        print(f"Observations: {len(result)}")
        print(f"Saved: {DATA_DIR / f'{market}_intrinsic_w2.txt'}")

    if not all_results:
        raise RuntimeError("No indices were successfully processed.")

    summary = create_summary(all_results)
    summary_file, table_file = save_summary(summary)

    print("\nAnalysis complete.")
    print(f"Summary: {summary_file}")
    print(f"Summary table: {table_file}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
