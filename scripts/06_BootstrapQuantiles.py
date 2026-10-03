"""
BootstrapQuantiles.py

Quantify uncertainty in the stress-to-normal ratio of quantile-specific
normalized Wasserstein distances.

Statistic:
    R_q = E[W2_q | stress] / E[W2_q | normal]

Algorithm:
- moving-block bootstrap with overlapping blocks;
- block length = 30 observations, matching the rolling W2 window;
- independent resampling of normal and stress regime series;
- 10,000 bootstrap replications;
- percentile 95% confidence intervals.

Input:
wasserstein_quantiles/{INDEX}_quantile_w2.txt

Output:
wasserstein_quantiles/quantile_w2_moving_block_bootstrap_30.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "wasserstein_quantiles"

OUTPUT_FILE = BASE_DIR / "quantile_w2_moving_block_bootstrap_30.txt"

INDICES = ["DAX", "WIG", "OMXT", "OMXR", "OMXV"]

QUANTILES = [
    "0_5pct",
    "5_25pct",
    "25_75pct",
    "75_95pct",
    "95_100pct",
]

N_BOOTSTRAP = 10_000
CONFIDENCE_LEVEL = 0.95
BLOCK_LENGTH = 30
RANDOM_SEED = 42


def load_quantile_w2(index_name):
    path = BASE_DIR / f"{index_name}_quantile_w2.txt"

    if not path.exists():
        raise FileNotFoundError(f"Quantile W2 file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "stress" not in df.columns:
        raise ValueError(
            f"{index_name}: no 'stress' column in {path}"
        )

    if df["stress"].dtype != bool:
        df["stress_bool"] = (
            df["stress"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "stress"])
        )
    else:
        df["stress_bool"] = df["stress"]

    quantile_columns = {}

    for q in QUANTILES:
        candidates = [
            f"w2_norm_{q}",
            f"w2_{q}",
            f"w2_normalized_{q}",
            f"normalized_w2_{q}",
            f"w2norm_{q}",
        ]

        found = next(
            (c for c in candidates if c in df.columns),
            None
        )

        if found is None:
            found = next(
                (
                    c for c in df.columns
                    if "w2" in c and q in c
                ),
                None
            )

        if found is None:
            raise ValueError(
                f"{index_name}: cannot find W2 column for {q}."
            )

        quantile_columns[q] = found

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )
        df = df.sort_values("date")

    return df, quantile_columns


def moving_block_sample(values, block_length, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < block_length:
        raise ValueError(
            f"Series length {n} is smaller than block length "
            f"{block_length}."
        )

    n_blocks = n - block_length + 1

    # Overlapping blocks.
    blocks = np.array([
        values[i:i + block_length]
        for i in range(n_blocks)
    ])

    n_required = int(np.ceil(n / block_length))

    selected = rng.integers(
        0, n_blocks, size=n_required
    )

    return blocks[selected].reshape(-1)[:n]


def moving_block_bootstrap_ratio(
    normal_values,
    stress_values,
    block_length,
    n_bootstrap,
    confidence_level,
    rng,
):
    normal_values = np.asarray(normal_values, dtype=float)
    stress_values = np.asarray(stress_values, dtype=float)

    normal_values = normal_values[np.isfinite(normal_values)]
    stress_values = stress_values[np.isfinite(stress_values)]

    if len(normal_values) < block_length:
        raise ValueError(
            f"Normal sample ({len(normal_values)}) is smaller "
            f"than block length ({block_length})."
        )

    if len(stress_values) < block_length:
        raise ValueError(
            f"Stress sample ({len(stress_values)}) is smaller "
            f"than block length ({block_length})."
        )

    observed_ratio = (
        np.mean(stress_values) /
        np.mean(normal_values)
    )

    bootstrap_ratios = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        normal_bootstrap = moving_block_sample(
            normal_values, block_length, rng
        )
        stress_bootstrap = moving_block_sample(
            stress_values, block_length, rng
        )

        bootstrap_ratios[b] = (
            np.mean(stress_bootstrap) /
            np.mean(normal_bootstrap)
        )

    alpha = 1.0 - confidence_level

    lower = np.percentile(
        bootstrap_ratios,
        100 * alpha / 2
    )
    upper = np.percentile(
        bootstrap_ratios,
        100 * (1 - alpha / 2)
    )

    return {
        "observed_ratio": observed_ratio,
        "bootstrap_mean": np.mean(bootstrap_ratios),
        "bootstrap_se": np.std(
            bootstrap_ratios, ddof=1
        ),
        "ci_lower": lower,
        "ci_upper": upper,
        "contains_one": lower <= 1.0 <= upper,
    }


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for index in INDICES:
        print("=" * 75)
        print(f"INDEX: {index}")

        try:
            df, quantile_columns = load_quantile_w2(index)

            normal_df = df[df["stress_bool"] == False]
            stress_df = df[df["stress_bool"] == True]

            print(f"Normal observations: {len(normal_df)}")
            print(f"Stress observations: {len(stress_df)}")

            for q in QUANTILES:
                column = quantile_columns[q]

                normal_values = pd.to_numeric(
                    normal_df[column], errors="coerce"
                ).dropna().to_numpy()

                stress_values = pd.to_numeric(
                    stress_df[column], errors="coerce"
                ).dropna().to_numpy()

                result = moving_block_bootstrap_ratio(
                    normal_values,
                    stress_values,
                    block_length=BLOCK_LENGTH,
                    n_bootstrap=N_BOOTSTRAP,
                    confidence_level=CONFIDENCE_LEVEL,
                    rng=rng,
                )

                results.append({
                    "index": index,
                    "quantile": q,
                    "normal_mean": np.mean(normal_values),
                    "normal_n": len(normal_values),
                    "stress_mean": np.mean(stress_values),
                    "stress_n": len(stress_values),
                    "R_stress_normal": result["observed_ratio"],
                    "bootstrap_mean": result["bootstrap_mean"],
                    "bootstrap_SE": result["bootstrap_se"],
                    "CI_lower": result["ci_lower"],
                    "CI_upper": result["ci_upper"],
                    "contains_R_equals_1": result["contains_one"],
                    "block_length": BLOCK_LENGTH,
                    "n_bootstrap": N_BOOTSTRAP,
                    "confidence_level": CONFIDENCE_LEVEL,
                })

                print(
                    f"{q:>10s} | "
                    f"R = {result['observed_ratio']:.4f} | "
                    f"95% CI = "
                    f"[{result['ci_lower']:.4f}, "
                    f"{result['ci_upper']:.4f}] | "
                    f"R=1 inside CI: "
                    f"{result['contains_one']}"
                )

        except Exception as exc:
            print(f"ERROR for {index}: {exc}")

    if not results:
        raise RuntimeError("No bootstrap results were produced.")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        float_format="%.8f"
    )

    print("=" * 75)
    print("MOVING-BLOCK BOOTSTRAP FINISHED")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
