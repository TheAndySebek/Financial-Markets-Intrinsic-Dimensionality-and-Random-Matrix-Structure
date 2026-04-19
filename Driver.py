import pandas as pd
import numpy as np
import SP500MarketStructure
# ------------------------------------------------------------
# 1. Instantiate class
# ------------------------------------------------------------
csv_path = "S&P 500 Historical Components & Changes(01-17-2026).csv"
model = SP500MarketStructure.SP500MarketStructure(csv_path)

# ------------------------------------------------------------
# 2. One-period analysis
# ------------------------------------------------------------
result = model.analyze(
    as_of_date="2020-01-02",
    start_date="2000-01-01",
    end_date="2020-01-02",
    return_method="log",
    min_nonmissing_fraction=0.80,
    k_neighbors=10,
    max_assets=300   # use None for all, but 300 is safer/faster while developing
)

# ------------------------------------------------------------
# 3. Summary output
# ------------------------------------------------------------
summary = model.summarize_result(result)
print("\n=== Summary ===")
print(summary)

print("\n=== RMT Result ===")
print(result.rmt_result)

print("\n=== First 10 Eigenvalues ===")
print(result.eigenvalues[:10])

# ------------------------------------------------------------
# 4. Plot eigenvalue spectrum and explained variance
# ------------------------------------------------------------
model.plot_eigenvalue_spectrum(result)
model.plot_explained_variance(result)

# ------------------------------------------------------------
# 5. Build cleaned covariance matrix
# ------------------------------------------------------------
cleaned_cov = model.rmt_clean_covariance(
    covariance_matrix=result.covariance_matrix,
    eigenvalues=result.eigenvalues,
    eigenvectors=result.eigenvectors,
    rmt_result=result.rmt_result
)

print("\n=== Cleaned Covariance Shape ===")
print(cleaned_cov.shape)

# ------------------------------------------------------------
# 6. Extract top eigenportfolio
# ------------------------------------------------------------
pc1_portfolio = model.eigenportfolio(result, component=0, normalize="l1")
print("\n=== Top 20 weights of first eigenportfolio by absolute magnitude ===")
print(pc1_portfolio.reindex(pc1_portfolio.abs().sort_values(ascending=False).index).head(20))

# ------------------------------------------------------------
# 7. Rolling dimension analysis
# ------------------------------------------------------------
rolling_df = model.rolling_dimension_analysis(
    as_of_date="2020-01-02",
    full_start_date="2017-01-01",
    full_end_date="2020-01-02",
    window_size=252,   # ~1 trading year
    step_size=21,      # ~1 trading month
    return_method="log",
    min_nonmissing_fraction=0.80,
    max_assets=250
)

print("\n=== Rolling Dimension Head ===")
print(rolling_df.head())

model.plot_rolling_dimensions(rolling_df)