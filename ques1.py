import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, WLS
from tabulate import tabulate
from skmisc import loess

# Define the data
Y = np.array([1, 0, 1, 4, 3, 2, 5, 6, 9, 13, 15, 16])
X = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [5, 4], [5, 6], [6, 5], [7, 4], [10, 8], [11, 7], [11, 9], [12, 10]])

# Add a constant (intercept) for statsmodels
X_with_intercept = sm.add_constant(X)

# OLS model
ols_model = OLS(Y, X_with_intercept)
ols_results = ols_model.fit()

# Printing OLS Results
print("OLS Results:")
print(f"{'Parameter':<12} {'Estimate':<10} {'Std. Error':<12} {'t value':<9} {'Pr(>|t|)':<10}")
for i, param in enumerate(ols_results.params):
    print(f"{ols_results.model.exog_names[i]:<12} {param:<10.4f} {ols_results.bse[i]:<12.4f} {ols_results.tvalues[i]:<9.4f} {ols_results.pvalues[i]:<10.4f}")

print(f"\nR-squared: {ols_results.rsquared:.4f}, Adjusted R-squared: {ols_results.rsquared_adj:.4f}")
print(f"F-statistic: {ols_results.fvalue:.2f} on {ols_results.df_model:.0f} and {ols_results.df_resid:.0f} DF, p-value: {ols_results.f_pvalue:.4e}\n")

# LOESS for WLS weights
loess_model = loess.loess(X[:, 0], Y, span=0.90, degree=2)
loess_model.fit()
fitted_values = loess_model.predict(X[:, 0]).values  # Extract fitted values correctly

# Calculate residuals and weights for WLS
residuals = Y - fitted_values
weights = 1 / (residuals**2 + 1e-8)  # Adding small constant to avoid division by zero

# WLS model
wls_model = WLS(Y, X_with_intercept, weights=weights)
wls_results = wls_model.fit()

# Printing WLS Results
print("\nWLS Results:")
print(f"{'Parameter':<12} {'Estimate':<10} {'Std. Error':<12} {'t value':<9} {'Pr(>|t|)':<10}")
for i, param in enumerate(wls_results.params):
    print(f"{wls_results.model.exog_names[i]:<12} {param:<10.4f} {wls_results.bse[i]:<12.4f} {wls_results.tvalues[i]:<9.4f} {wls_results.pvalues[i]:<10.4f}")

print(f"\nR-squared: {wls_results.rsquared:.4f}, Adjusted R-squared: {wls_results.rsquared_adj:.4f}")
print(f"F-statistic: {wls_results.fvalue:.1f} on {wls_results.df_model:.1f} and {wls_results.df_resid:.1f} DF, p-value: {wls_results.f_pvalue:.4e}")
