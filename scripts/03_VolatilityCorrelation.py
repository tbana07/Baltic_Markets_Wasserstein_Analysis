"""
VolatilityCorrelation.py

Study the relationship between normalized intrinsic W_2 and 30-day
annualized volatility for DAX, WIG, OMXT, OMXR and OMXV.

Algorithms:
- Pearson correlation;
- Spearman rank correlation;
- distance correlation;
- ordinary least-squares linear regression and R^2;
- correlations of first differences;
- LOWESS nonparametric smoothing for visualization;
- market-specific 90th-percentile volatility stress classification.

The script uses the precomputed files produced by BaseIndicators.py and
IntrinsicW2.py and aligns them by date.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
W2_DIR = PROJECT_ROOT / "intrinsic_wasserstein" / "data"
VOLATILITY_DIR = PROJECT_ROOT / "base_indicators"
OUTPUT_DIR = PROJECT_ROOT / "wasserstein_correlation_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"

MARKETS = ["DAX", "WIG", "OMXV", "OMXR", "OMXT"]
W2_COLUMN = "w2_normalized"
VOLATILITY_COLUMN = "volatility_30d"


def distance_correlation(x, y):
    try:
        import dcor
        return float(dcor.distance_correlation(x, y))
    except ImportError:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        def centered_distance_matrix(z):
            D = np.abs(z[:, None] - z[None, :])
            return (
                D
                - D.mean(axis=1, keepdims=True)
                - D.mean(axis=0, keepdims=True)
                + D.mean()
            )

        A = centered_distance_matrix(x)
        B = centered_distance_matrix(y)

        dcov2 = np.mean(A * B)
        dvarx = np.mean(A * A)
        dvary = np.mean(B * B)
        denom = np.sqrt(dvarx * dvary)

        if denom <= 0:
            return np.nan

        return float(np.sqrt(max(dcov2, 0.0) / denom))


def load_existing_series(index):
    w2_path = W2_DIR / f"{index}_intrinsic_w2.txt"
    volatility_path = (
        VOLATILITY_DIR /
        f"{index}_indicators" /
        "index_with_indicators.txt"
    )

    if not w2_path.exists():
        raise FileNotFoundError(f"W2 file not found: {w2_path}")
    if not volatility_path.exists():
        raise FileNotFoundError(
            f"Volatility file not found: {volatility_path}"
        )

    w2 = pd.read_csv(w2_path, sep="\t")
    volatility = pd.read_csv(volatility_path, sep=",")

    w2.columns = [str(c).strip() for c in w2.columns]
    volatility.columns = [str(c).strip() for c in volatility.columns]

    required_w2 = ["date", W2_COLUMN]
    required_vol = ["date", VOLATILITY_COLUMN]

    missing = [c for c in required_w2 if c not in w2.columns]
    if missing:
        raise ValueError(f"{index}: W2 columns missing: {missing}")

    missing = [c for c in required_vol if c not in volatility.columns]
    if missing:
        raise ValueError(
            f"{index}: volatility columns missing: {missing}"
        )

    w2 = w2[required_w2].copy()
    volatility = volatility[required_vol].copy()

    w2.columns = ["date", "W2_normalized"]
    volatility.columns = ["date", "volatility"]

    for df in (w2, volatility):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    w2["W2_normalized"] = pd.to_numeric(
        w2["W2_normalized"], errors="coerce"
    )
    volatility["volatility"] = pd.to_numeric(
        volatility["volatility"], errors="coerce"
    )

    return (
        pd.merge(w2, volatility, on="date", how="inner")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["W2_normalized", "volatility"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def safe_pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return tuple(float(v) for v in pearsonr(x, y))


def safe_spearman(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    return tuple(float(v) for v in spearmanr(x, y))


def linear_regression_stats(x, y):
    model = LinearRegression().fit(
        np.asarray(x).reshape(-1, 1),
        np.asarray(y)
    )
    return {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "r2": float(model.score(
            np.asarray(x).reshape(-1, 1),
            np.asarray(y)
        )),
    }


def analyze_series(df):
    x = df["volatility"].to_numpy()
    y = df["W2_normalized"].to_numpy()

    pearson, pearson_p = safe_pearson(x, y)
    spearman, spearman_p = safe_spearman(x, y)

    reg = linear_regression_stats(x, y)

    dx = np.diff(x)
    dy = np.diff(y)

    diff_pearson, diff_pearson_p = safe_pearson(dx, dy)
    diff_spearman, diff_spearman_p = safe_spearman(dx, dy)

    return {
        "n": len(df),
        "pearson_r": pearson,
        "pearson_p": pearson_p,
        "spearman_rho": spearman,
        "spearman_p": spearman_p,
        "distance_correlation": distance_correlation(x, y),
        "regression_intercept": reg["intercept"],
        "regression_slope": reg["slope"],
        "linear_R2": reg["r2"],
        "diff_pearson_r": diff_pearson,
        "diff_pearson_p": diff_pearson_p,
        "diff_spearman_rho": diff_spearman,
        "diff_spearman_p": diff_spearman_p,
        "diff_distance_correlation": distance_correlation(dx, dy),
    }


def plot_scatter(df, index, stats):
    x = df["volatility"].to_numpy()
    y = df["W2_normalized"].to_numpy()
    stress = x >= np.quantile(x, 0.90)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    ax.scatter(
        x[~stress], y[~stress], s=10, alpha=0.28,
        label="Normal (< 90th percentile)"
    )
    ax.scatter(
        x[stress], y[stress], s=14, alpha=0.50,
        label="Stress (≥ 90th percentile)"
    )

    xx = np.linspace(x.min(), x.max(), 300)
    yy = (
        stats["regression_intercept"] +
        stats["regression_slope"] * xx
    )
    ax.plot(xx, yy, linewidth=2, label="OLS fit")

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(
            y, x, frac=0.15, it=1, return_sorted=True
        )
        ax.plot(
            smoothed[:, 0], smoothed[:, 1],
            linewidth=2, linestyle="--", label="LOWESS"
        )
    except ImportError:
        pass

    ax.set_xlabel(r"Volatility $\sigma$")
    ax.set_ylabel(r"Normalized Wasserstein $W_2$")
    ax.set_title(f"{index}: volatility vs normalized Wasserstein $W_2$")
    ax.grid(alpha=0.20)
    ax.legend()

    text = (
        f"Pearson r = {stats['pearson_r']:.4f}\n"
        f"Spearman ρ = {stats['spearman_rho']:.4f}\n"
        f"Distance corr. = {stats['distance_correlation']:.4f}\n"
        f"Linear $R^2$ = {stats['linear_R2']:.4f}"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / f"{index}_sigma_vs_W2_normalized.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for index in MARKETS:
        try:
            df = load_existing_series(index)
            stats = analyze_series(df)
            stats["market"] = index
            results.append(stats)
            plot_scatter(df, index, stats)

            print(
                f"{index}: n={stats['n']}, "
                f"Pearson={stats['pearson_r']:.4f}, "
                f"Spearman={stats['spearman_rho']:.4f}, "
                f"dCor={stats['distance_correlation']:.4f}, "
                f"R2={stats['linear_R2']:.4f}"
            )
        except Exception as exc:
            print(f"[WARNING] {index}: {exc}")

    if not results:
        raise RuntimeError("No datasets could be analyzed.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "correlation_results.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "correlation_results.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("NORMALIZED WASSERSTEIN W2 vs VOLATILITY\n")
        f.write("=" * 72 + "\n\n")
        f.write(
            "Input series are precomputed normalized W2 and "
            "30-day annualized volatility.\n\n"
        )
        f.write(
            "Because these are rolling-window series, adjacent "
            "observations are serially dependent; ordinary iid "
            "correlation p-values are descriptive rather than "
            "a final time-series inference procedure.\n\n"
        )

        for _, row in results_df.iterrows():
            f.write("-" * 72 + "\n")
            f.write(f"MARKET: {row['market']}\n")
            f.write(f"Observations: {int(row['n'])}\n\n")
            f.write("LEVEL RELATIONSHIP\n")
            f.write(f"Pearson r: {row['pearson_r']:.8f}\n")
            f.write(f"Pearson p-value: {row['pearson_p']:.8g}\n")
            f.write(f"Spearman rho: {row['spearman_rho']:.8f}\n")
            f.write(f"Spearman p-value: {row['spearman_p']:.8g}\n")
            f.write(
                f"Distance correlation: "
                f"{row['distance_correlation']:.8f}\n"
            )
            f.write(
                f"OLS slope: {row['regression_slope']:.8f}\n"
            )
            f.write(
                f"OLS intercept: {row['regression_intercept']:.8f}\n"
            )
            f.write(
                f"Linear regression R^2: {row['linear_R2']:.8f}\n\n"
            )
            f.write("FIRST-DIFFERENCE RELATIONSHIP\n")
            f.write(
                f"Pearson r [Δ]: {row['diff_pearson_r']:.8f}\n"
            )
            f.write(
                f"Spearman rho [Δ]: "
                f"{row['diff_spearman_rho']:.8f}\n"
            )
            f.write(
                f"Distance correlation [Δ]: "
                f"{row['diff_distance_correlation']:.8f}\n\n"
            )

    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
