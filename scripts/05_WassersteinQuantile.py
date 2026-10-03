"""
WassersteinQuantile.py

Measure where in the return distribution changes in normalized
Wasserstein distance are concentrated.

Algorithms:
- 30-day rolling standardized-return windows;
- five empirical quantile bands:
  0–5%, 5–25%, 25–75%, 75–95%, 95–100%;
- one-dimensional empirical W_2 within corresponding bands;
- stress/normal classification inherited from the intrinsic W_2 file;
- conditional expected W_2 for normal and stress observations.

Outputs:
wasserstein_quantiles/{INDEX}_quantile_w2.txt
wasserstein_quantiles/{INDEX}_quantile_expected_values.txt
and a combined expected-values table.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INDICATORS = PROJECT_ROOT / "base_indicators"
BASE_WASSERSTEIN = PROJECT_ROOT / "intrinsic_wasserstein" / "data"
OUTPUT_DIR = PROJECT_ROOT / "wasserstein_quantiles"

INDICES = ["DAX", "WIG", "OMXT", "OMXR", "OMXV"]

ROLLING_WINDOW = 30

QUANTILE_BANDS = [
    (0.00, 0.05),
    (0.05, 0.25),
    (0.25, 0.75),
    (0.75, 0.95),
    (0.95, 1.00),
]

QUANTILE_NAMES = [
    "0-5%",
    "5-25%",
    "25-75%",
    "75-95%",
    "95-100%",
]


def column_name(name):
    return "W2_norm_" + name.replace("%", "pct").replace("-", "_")


def load_z_scores(index_name):
    path = (
        BASE_INDICATORS /
        f"{index_name}_indicators" /
        "index_z_scores.txt"
    )

    if not path.exists():
        raise FileNotFoundError(f"Z-score file not found: {path}")

    df = pd.read_csv(path, sep=",")
    df.columns = [str(c).strip() for c in df.columns]

    required = ["date", "z_score_30d"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{index_name}: missing z-score columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["z_score_30d"] = pd.to_numeric(
        df["z_score_30d"], errors="coerce"
    )

    return (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date")
        [required]
        .reset_index(drop=True)
    )


def load_stress(index_name):
    path = BASE_WASSERSTEIN / f"{index_name}_intrinsic_w2.txt"

    if not path.exists():
        raise FileNotFoundError(f"Wasserstein file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    required = ["date", "stress"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{index_name}: missing stress columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["stress"].dtype != bool:
        df["stress"] = (
            df["stress"].astype(str).str.strip().str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    return (
        df.dropna(subset=required)
        .drop_duplicates("date")
        [required]
        .reset_index(drop=True)
    )


def get_quantile_slice(values, q_low, q_high):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)

    if n == 0:
        return np.array([])

    lower = int(np.floor(q_low * n))
    upper = int(np.floor(q_high * n))

    lower = max(0, min(lower, n - 1))
    upper = max(lower + 1, min(upper, n))

    return values[lower:upper]


def wasserstein_w2(sample_a, sample_b):
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    if len(a) == len(b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    n = max(len(a), len(b))
    probabilities = (np.arange(n) + 0.5) / n

    def empirical_quantile(sample, probabilities):
        m = len(sample)
        sample_probabilities = (np.arange(m) + 0.5) / m
        return np.interp(
            probabilities,
            sample_probabilities,
            sample
        )

    qa = empirical_quantile(a, probabilities)
    qb = empirical_quantile(b, probabilities)

    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def calculate_quantile_w2(index_name):
    zscores = load_z_scores(index_name)
    stress = load_stress(index_name)

    df = (
        pd.merge(zscores, stress, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df) < ROLLING_WINDOW + 1:
        raise ValueError(
            f"{index_name}: too few observations: {len(df)}"
        )

    result = pd.DataFrame({
        "date": df["date"],
        "stress": df["stress"],
    })

    for name in QUANTILE_NAMES:
        result[column_name(name)] = np.nan

    values = df["z_score_30d"].to_numpy()

    for t in range(ROLLING_WINDOW, len(df)):
        previous_window = values[t - ROLLING_WINDOW:t]
        current_window = values[t - ROLLING_WINDOW + 1:t + 1]

        for (q_low, q_high), name in zip(
            QUANTILE_BANDS, QUANTILE_NAMES
        ):
            previous_quantile = get_quantile_slice(
                previous_window, q_low, q_high
            )
            current_quantile = get_quantile_slice(
                current_window, q_low, q_high
            )

            result.loc[t, column_name(name)] = wasserstein_w2(
                previous_quantile,
                current_quantile
            )

    return result


def calculate_expected_values(result):
    rows = []

    for name in QUANTILE_NAMES:
        col = column_name(name)

        normal = result.loc[
            result["stress"] == False, col
        ].dropna()

        stress = result.loc[
            result["stress"] == True, col
        ].dropna()

        rows.append({
            "quantile": name,
            "normal_mean": normal.mean(),
            "normal_n": len(normal),
            "stress_mean": stress.mean(),
            "stress_n": len(stress),
        })

    return pd.DataFrame(rows)


def save_rolling_results(index_name, result):
    path = OUTPUT_DIR / f"{index_name}_quantile_w2.txt"
    result.to_csv(
        path,
        sep="\t",
        index=False,
        float_format="%.10f"
    )
    return path


def save_expected_values(index_name, expected):
    path = OUTPUT_DIR / f"{index_name}_quantile_expected_values.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"Expected normalized W2 by quantile and volatility "
            f"regime: {index_name}\n"
        )
        f.write("=" * 90 + "\n\n")
        f.write(
            "0-5% = lower tail\n"
            "5-25% = lower-central region\n"
            "25-75% = central region\n"
            "75-95% = upper-central region\n"
            "95-100% = upper tail\n\n"
        )
        f.write(
            f"{'Quantile':<20}"
            f"{'Normal mean':>18}"
            f"{'Normal N':>12}"
            f"{'Stress mean':>18}"
            f"{'Stress N':>12}\n"
        )
        f.write("-" * 90 + "\n")

        for _, row in expected.iterrows():
            f.write(
                f"{row['quantile']:<20}"
                f"{row['normal_mean']:>18.10f}"
                f"{row['normal_n']:>12d}"
                f"{row['stress_mean']:>18.10f}"
                f"{row['stress_n']:>12d}\n"
            )

    return path


def plot_quantile_w2(index_name, result):
    fig, ax = plt.subplots(figsize=(14, 7))

    for name in QUANTILE_NAMES:
        ax.plot(
            result["date"],
            result[column_name(name)],
            label=name,
            linewidth=1.0
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized $W_2$")
    ax.set_title(
        f"{index_name}: rolling normalized $W_2$ "
        "by distribution quantile"
    )
    ax.legend(title="Quantile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = OUTPUT_DIR / f"{index_name}_quantile_w2.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_expected = []

    for index_name in INDICES:
        print("=" * 80)
        print(f"Processing {index_name}")

        try:
            result = calculate_quantile_w2(index_name)
            rolling_path = save_rolling_results(index_name, result)

            expected = calculate_expected_values(result)
            expected_path = save_expected_values(index_name, expected)

            plot_path = plot_quantile_w2(index_name, result)

            expected_copy = expected.copy()
            expected_copy.insert(0, "index", index_name)
            all_expected.append(expected_copy)

            print(f"Rolling results: {rolling_path}")
            print(f"Expected values: {expected_path}")
            print(f"Plot: {plot_path}")

        except Exception as exc:
            print(f"ERROR for {index_name}: {exc}")

    if all_expected:
        combined = pd.concat(all_expected, ignore_index=True)
        combined_path = (
            OUTPUT_DIR / "all_indices_quantile_expected_values.txt"
        )
        combined.to_csv(
            combined_path,
            sep="\t",
            index=False,
            float_format="%.10f"
        )
        print(f"Combined expected values: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
