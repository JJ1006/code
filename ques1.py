import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS
from skmisc.loess import loess
import statsmodels.api as sm
from tabulate import tabulate

# Define the data
Y = np.array([1, 0, 1, 4, 3, 2, 5, 6, 9, 13, 15, 16])
X = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [5, 4], [5, 6], [6, 5], [7, 4], [10, 8], [11, 7], [11, 9], [12, 10]])

# Create a DataFrame
data = pd.DataFrame(X, columns=['X1', 'X2'])
data['Y'] = Y

# Perform OLS
X_with_const = sm.add_constant(X)
model_ols = OLS(Y, X_with_const).fit()

# Perform LOESS to get weights
loess_model = loess(X, Y, span=0.90, degree=2)
loess_model.fit()
predicted = loess_model.predict(X).values
residuals = Y - predicted
weights = 1 / (residuals**2 + 1e-8)  # Adding small constant to avoid division by zero

# Perform WLS
model_wls = WLS(Y, X_with_const, weights=weights).fit()

def print_header(text):
    print("\n" + "=" * 60)
    print(f"{text:^60}")
    print("=" * 60 + "\n")

def print_model_summary(model, title):
    print_header(title)
    summary = model.summary()
    print(summary)

def print_comparison(ols_model, wls_model):
    print_header("Comparison of OLS and WLS Results")
    
    # Prepare data for tabulate
    data = []
    for i, param in enumerate(['Intercept', 'X1', 'X2']):
        data.append([
            param,
            f"{ols_model.params[i]:.4f}",
            f"{wls_model.params[i]:.4f}",
            f"{ols_model.pvalues[i]:.4f}",
            f"{wls_model.pvalues[i]:.4f}"
        ])
    
    headers = ["Parameter", "OLS Estimate", "WLS Estimate", "OLS p-value", "WLS p-value"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    print(f"OLS R-squared: {ols_model.rsquared:.4f}")

# Print results
print_model_summary(model_ols, "OLS Regression Results")
print_model_summary(model_wls, "WLS Regression Results")
print_comparison(model_ols, model_wls)
