import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import pandas as pd

# Set parameters
n = 10  # number of trials
p = 0.5  # probability of success

# Part 1: Create the table
def create_binomial_table(n, p):
    r_values = np.arange(0, n+1)
    probabilities = binom.pmf(r_values, n, p)
    cumulative_probabilities = binom.cdf(r_values, n, p)
    
    df = pd.DataFrame({
        'r': r_values,
        'p(r)': probabilities,
        'Cumulative P(r)': cumulative_probabilities
    })
    
    return df

# Part 2: Create cumulative distribution graph
def plot_cumulative_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['r'], df['Cumulative P(r)'], marker='o')
    plt.title('Cumulative Distribution Function')
    plt.xlabel('r')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.show()

# Part 3: Create discrete histogram
def plot_probability_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df['r'], df['p(r)'], alpha=0.8, color='skyblue', edgecolor='navy')
    plt.title('Binomial Probability Distribution')
    plt.xlabel('r')
    plt.ylabel('Probability')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Execute all parts
table = create_binomial_table(n, p)
print("Binomial Probability Distribution Table:")
print(table.to_string(index=False))

plot_cumulative_distribution(table)
plot_probability_distribution(table)
