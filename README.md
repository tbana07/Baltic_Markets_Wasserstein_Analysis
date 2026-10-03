# Baltic Stock Markets Distributional-Shape Homogenity - a Wasserstein distance analysis

This project aims to investigate whether Baltic stock markets exhibit
homogeneous dynamics in the shape of their return distributions. The analysis applies
the 2-Wasserstein distance to rolling normalized empirical distributions of daily returns
for the Tallinn (OMXT), Riga (OMXR), and Vilnius (OMXV) stock indices, with the
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
