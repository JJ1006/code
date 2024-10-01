import numpy as np

# data already given
Y = np.array([1, 0, 1, 4, 3, 2, 5, 6, 9, 13, 15, 16])
X = np.array([
    [1,  1,  1],
    [1,  2,  1],
    [1,  2,  2],
    [1,  3,  2],
    [1,  5,  4],
    [1,  5,  6],
    [1,  6,  5],
    [1,  7,  4],
    [1, 10,  8],
    [1, 11,  7],
    [1, 11,  9],
    [1, 12, 10]
])

# Extracting the predictors X1 and X2 (excluding the intercept term)
X_predictors = X[:, 1:]

# Calculating the covariance matrix
cov_matrix = np.cov(X_predictors.T)

# Calculating correlation matrix for comparison
corr_matrix = np.corrcoef(X_predictors.T)

def print_matrix(title, matrix):
    print(f"\n{title}")
    print("=" * len(title))
    print(np.array2string(matrix, precision=4, suppress_small=True))

print_matrix("The Covariance Matrix for predictors X1 and X2", cov_matrix)
print_matrix("The Correlation Matrix for predictors X1 and X2", corr_matrix)
