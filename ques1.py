import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from tabulate import tabulate

# Define the data
Y = np.array([1, 0, 1, 4, 3, 2, 5, 6, 9, 13, 15, 16])
X = np.array([[1, 1, 1], [1, 2, 1], [1, 2, 2], [1, 3, 2], [1, 5, 4], [1, 5, 6],
              [1, 6, 5], [1, 7, 4], [1, 10, 8], [1, 11, 7], [1, 11, 9], [1, 12, 10]])

# Perform OLS
model_ols = OLS(Y, add_constant(X[:, 1:])).fit()

# Perform LOWESS to get weights (dummy weights used here for illustration)
predicted_ols = model_ols.predict()
residuals_ols = Y - predicted_ols
weights = 1 / (residuals_ols**2 + 1e-8)  # Adding small constant to avoid division by zero

# Perform WLS
model_wls = WLS(Y, add_constant(X[:, 1:]), weights=weights).fit()

def print_model_summary(model, model_type):
    print(f"{model_type} Results:")
    
    # Prepare data for tabulate
    data = []
    for i, param in enumerate(['Intercept', 'X1', 'X2']):
        data.append([
            param,
            f"{model.params[i]:.4f}",
            f"{model.bse[i]:.4f}",
            f"{model.tvalues[i]:.4f}",
            f"{model.pvalues[i]:.4f}"
        ])
    
    headers = ["Parameter", "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    print(f"R-squared: {model.rsquared:.4f}, Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.2f} on {model.df_model:.0f} and {model.df_resid:.0f} DF, p-value: {model.f_pvalue:.3e}")

# Print results
print_model_summary(model_ols, "OLS")
print()  # Add an empty line between OLS and WLS results
print_model_summary(model_wls, "WLS")
