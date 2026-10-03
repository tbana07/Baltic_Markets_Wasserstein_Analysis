"""
IndicatorsCorrelation.py

Fit a multivariate model explaining normalized intrinsic W_2 with
standard distributional indicators.

Model:
    W2_normalized(t) =
        b0 + b1*sigma(t) + b2*S(t) + b3*K(t) + epsilon(t)

Algorithms:
- ordinary least squares;
- volatility, skewness and excess kurtosis as regressors;
- Newey-West HAC covariance for time-series-robust standard errors;
- automatic bandwidth L = floor(4*(T/100)^(2/9));
- R^2, adjusted R^2, HAC Wald/F statistics and confidence intervals.

The script processes all five indices using repository-relative paths.
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INDICATORS = PROJECT_ROOT / "base_indicators"
BASE_WASSERSTEIN = PROJECT_ROOT / "intrinsic_wasserstein" / "data"
OUTPUT_DIR = PROJECT_ROOT / "wasserstein_correlation_analysis"

INDICES = ["DAX", "WIG", "OMXT", "OMXR", "OMXV"]


def load_indicators(index_name):
    path = (
        BASE_INDICATORS /
        f"{index_name}_indicators" /
        "index_with_indicators.txt"
    )

    if not path.exists():
        raise FileNotFoundError(f"Indicator file not found: {path}")

    df = pd.read_csv(path, sep=",")
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "date",
        "volatility_30d_annualized",
        "skewness_30d",
        "kurtosis_excess_30d",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{index_name}: missing indicator columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[required]


def load_wasserstein(index_name):
    path = BASE_WASSERSTEIN / f"{index_name}_intrinsic_w2.txt"

    if not path.exists():
        raise FileNotFoundError(f"Wasserstein file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    required = ["date", "w2_normalized"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{index_name}: missing Wasserstein columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[required]


def newey_west_lag_length(n):
    lag = int(4 * (n / 100.0) ** (2.0 / 9.0))
    return max(0, min(lag, n - 1))


def fit_model(index_name):
    indicators = load_indicators(index_name)
    wasserstein = load_wasserstein(index_name)

    df = pd.merge(
        indicators,
        wasserstein,
        on="date",
        how="inner"
    )

    df = (
        df.replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df) < 10:
        raise ValueError(
            f"{index_name}: too few observations: {len(df)}"
        )

    y = df["w2_normalized"]
    X = sm.add_constant(
        df[
            [
                "volatility_30d_annualized",
                "skewness_30d",
                "kurtosis_excess_30d",
            ]
        ]
    )

    ols_model = sm.OLS(y, X).fit()
    hac_lags = newey_west_lag_length(len(df))

    hac_model = ols_model.get_robustcov_results(
        cov_type="HAC",
        maxlags=hac_lags,
        use_correction=True
    )

    return df, ols_model, hac_model, hac_lags


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "indicators_correlation.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "Regression of normalized Wasserstein distance "
            "on volatility, skewness and kurtosis\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write(
            "Model:\n"
            "W2_normalized(t) = b0 + b1*sigma(t) "
            "+ b2*S(t) + b3*K(t) + epsilon(t)\n\n"
        )
        f.write(
            "Inference: OLS coefficients with Newey-West HAC "
            "standard errors.\n"
        )
        f.write(
            "Automatic HAC lag: L = floor(4 * (T / 100)^(2/9)).\n\n"
        )

        for index_name in INDICES:
            print(f"Processing {index_name}...")

            try:
                df, ols_model, model, hac_lags = fit_model(index_name)
            except Exception as exc:
                print(f"ERROR for {index_name}: {exc}")
                f.write(
                    f"\n{'=' * 80}\n{index_name}\n{'=' * 80}\n"
                    f"ERROR:\n{exc}\n"
                )
                continue

            f.write(
                f"\n{'=' * 80}\n"
                f"{index_name}\n"
                f"{'=' * 80}\n\n"
            )
            f.write(f"Number of observations: {len(df)}\n")
            f.write(
                f"Date range: {df['date'].min().date()} "
                f"to {df['date'].max().date()}\n"
            )
            f.write(f"Newey-West HAC lag length: {hac_lags}\n\n")

            f.write(
                f"R-squared:          {ols_model.rsquared:.6f}\n"
                f"Adjusted R-squared: {ols_model.rsquared_adj:.6f}\n"
            )
            f.write(
                f"HAC F-statistic:         {model.fvalue:.6f}\n"
                f"HAC F-statistic p-value: {model.f_pvalue:.6e}\n\n"
            )

            f.write("Coefficients with Newey-West HAC standard errors\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Variable':<30}"
                f"{'Coefficient':>15}"
                f"{'HAC Std. Error':>18}"
                f"{'HAC t-stat':>15}"
                f"{'HAC p-value':>15}\n"
            )

            param_names = ols_model.params.index

            for i, variable in enumerate(param_names):
                f.write(
                    f"{variable:<30}"
                    f"{model.params[i]:>15.8f}"
                    f"{model.bse[i]:>18.8f}"
                    f"{model.tvalues[i]:>15.6f}"
                    f"{model.pvalues[i]:>15.6e}\n"
                )

            f.write("\n95% Confidence Intervals — Newey-West HAC\n")
            f.write("-" * 80 + "\n")

            conf = model.conf_int()
            for i, variable in enumerate(param_names):
                f.write(
                    f"{variable:<30}"
                    f"[{conf[i, 0]:.8f}, {conf[i, 1]:.8f}]\n"
                )

            residuals = ols_model.resid
            f.write("\nResidual statistics\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean residual: {residuals.mean():.8f}\n")
            f.write(f"Std. residual: {residuals.std():.8f}\n")
            f.write(f"Min residual: {residuals.min():.8f}\n")
            f.write(f"Max residual: {residuals.max():.8f}\n\n")

            f.write("Full OLS summary with Newey-West HAC inference\n")
            f.write("-" * 80 + "\n\n")
            f.write(model.summary().as_text())
            f.write("\n\n")

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
