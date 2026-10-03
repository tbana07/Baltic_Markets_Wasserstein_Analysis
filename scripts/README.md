# Project scripts

Run the scripts from this directory (or from the repository root using
the corresponding relative path). All scripts determine the repository
root automatically with:

    Path(__file__).resolve().parents[1]

Expected repository structure:

    data/raw/
    base_indicators/
    intrinsic_wasserstein/data/
    intrinsic_wasserstein/plots/
    wasserstein_quantiles/
    wasserstein_correlation_analysis/
    scripts/

The scripts process these indices:

    DAX, WIG, OMXT, OMXR, OMXV

Recommended order:

1. 01_BaseIndicators.py
2. 02_IntrinsicW2.py
3. 03_VolatilityCorrelation.py
4. 04_IndicatorsCorrelation.py
5. 05_WassersteinQuantile.py
6. 06_BootstrapQuantiles.py
