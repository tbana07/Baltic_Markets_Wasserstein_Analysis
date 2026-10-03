# Baltic Stock Markets Distributional-Shape Homogeneity - a Wasserstein distance analysis

This repository contains the computational framework and results for a study of distributional similarity and shape changes in Baltic equity markets using the Wasserstein distance. (For full paper see, [*Baltic Stock Markets Distributional-Shape Homogeneity - a Wasserstein distance analysis*](https://github.com/tbana07/Wasserstein_Baltic/blob/main/Baltic_Shape_Homogeneity.pdf))

Project aims to investigate whether Baltic stock markets exhibit
homogeneous dynamics in the shape of their return distributions. The analysis applies
the 2-Wasserstein distance to rolling normalized empirical distributions of daily returns
for the Tallinn (**OMXT**), Riga (**OMXR**), and Vilnius (**OMXV**) stock indices, with the
DAX and WIG serving as larger-market comparators. Using data from January 2012
to December 2025, distributional changes are examined both in relation to volatility
and across different parts of the return distribution. The results reveal substantial
heterogeneity within the Baltic region.

## Questions asked

* Are the Baltic stock markets homogeneous with respect to changes in the shape of their
normalized return distributions?
* To what extent is the change in distributional shape, measured by the normalized W2 distance,
related to conventional measures of volatility, skewness, and kurtosis?
* Does the relationship between volatility and distributional-shape change differ between Tallinn,
Riga, and Vilnius?
* Do Baltic markets exhibit a distributional response to volatility stress that is distinct from
that observed in larger European markets represented by the DAX and WIG?
* How does the concentration of changes in distributional shape in particular parts of the
return distribution differ betweeen the Baltic and larger Europen markets?

## Methodology

For 30-observation rolling windows daily logarithmic returns are calculated and normalized. Tha for two consecutive empirical distributions (P) and (Q), the one-dimensional second-order Wasserstein distance (W_2)
is comupted. 

Than rolling W_2 is compared with volatility and other conventional moments using Pearson, Spearman, Distance Correlation and OLS. Each markest is studied both in normall and stress conditions.

![Alt text](https://github.com/tbana07/Wasserstein_Baltic/blob/main/wasserstein_correlation_analysis/plots/AdjustedR2ofW_2Reg.png)

This is followed by analysis of shape changes in stress and normal regimes in each quantile of the distribution for each index. To adress the issue of small number of observations and temporal depandance between returns in consecutive rolling windows Moving-Block Bootstrap (MBB) is used. Results show large disproportions of change in W_2 between stress and normal situation in each quantile among Baltic Markets.

![Alt text](https://github.com/tbana07/Wasserstein_Baltic/blob/main/wasserstein_quantiles/plots/W2_quantile_ratio_bootstrap.png)

## Results

Tallinn exhibits a weak relationship between
normalized Wasserstein distance and volatility, closely resembling the behaviour of
DAX and WIG. In contrast, Riga and Vilnius show substantially stronger associations,
particularly during high-volatility periods. A multivariate analysis including volatility,
skewness, and kurtosis leads to the same distinction. Quantile-level analysis further
shows that stress-related changes in distributional shape differ considerably across the
three Baltic markets. Overall, the results do not support treating Baltic markets as
homogeneous with respect to distributional-shape dynamics. Instead, Tallinn displays
behaviour closer to the larger reference markets, while Riga and Vilnius exhibit a
distinct and more volatility-related pattern.

## Technicals

### Data 

The analysis uses daily index data obtained from Stooq.com.

The main sample covers approximately:
*January 2012 – December 2025
*Daily observations
*Rolling window: 30 trading days

The three Baltic indices are relatively similar in their concentration among large constituents. OMXT and OMXV are also relatively similar in total capitalization, while OMXR represents a substantially smaller market. These characteristics are considered when interpreting the results.

### Running

The scripts are designed to be executed sequentially:
1. `python scripts/01_BaseIndicators.py` - calculates rolling return statistics
2. `python scripts/02_IntrinsicW2.py` - calculates raw and normalized rolling (W_2) measures
3. `python scripts/03_VolatilityCorrelation.py` - analyses volatility and Wasserstein distance
4. `python scripts/04_IndicatorsCorrelation.py` - examines the relationship between normalized Wasserstein distance and volatility, skewness and kurtosis
5. `python scripts/05_WassersteinQuantile.py` - return distribution is divided into five quantile regions to identify which parts of the distribution contribute most strongly to Wasserstein differences
6. `python scripts/06_BootstrapQuantiles.py` - calculates stress-to-normal Wasserstein ratios and their moving-block bootstrap confidence intervals

## Interpretation

The Wasserstein framework is used here as a distributional comparison measure, rather than as a replacement for conventional financial indicators.


The distinction between raw and normalized Wasserstein distance is particularly important:

* raw (W_2) incorporates changes in location, scale and shape;
* normalized (W_2) suppresses location and scale differences and emphasizes changes in distributional shape.

This distinction allows the analysis to investigate whether observed similarities between markets arise primarily from common volatility dynamics or from genuinely similar distributional shapes.
