import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Given data
absences = [0, 3, 2, 2, 6, 0, 1, 0, 0, 1, 0, 2, 0, 1, 1, 0, 4, 0, 0, 4]
N = len(absences)

# Create a frequency table
unique, counts = np.unique(absences, return_counts=True)
freq_table = pd.DataFrame({
    'Score (x)': unique,
    'Freq': counts,
    'Prob p(x)': counts / N,
    'x * p(x)': unique * (counts / N),
    'x^2 * p(x)': (unique ** 2) * (counts / N)
})

# Calculate totals
totals = freq_table.sum().to_frame().T
totals['Score (x)'] = 'Total'
freq_table = pd.concat([freq_table, totals], ignore_index=True)

# Function to create a styled table
def create_styled_table(df):
    return tabulate(df, headers='keys', tablefmt='pretty', floatfmt='.4f', numalign='right')

# Print the styled table
print("Student Absence Data Table:")
print(create_styled_table(freq_table))
print()

# 1) Calculate the mean number of days lost
mean_days_lost = freq_table.loc[freq_table['Score (x)'] != 'Total', 'x * p(x)'].sum()
print(f"1) Mean number of days lost: {mean_days_lost:.4f}")

# 2) Calculate the variance of days lost
variance_days_lost = freq_table.loc[freq_table['Score (x)'] != 'Total', 'x^2 * p(x)'].sum() - mean_days_lost**2
print(f"2) Variance of days lost: {variance_days_lost:.4f}")

# Visualization: Histogram of absences
plt.figure(figsize=(10, 6))
plt.hist(absences, bins=range(min(absences), max(absences) + 2, 1), align='left', rwidth=0.8)
plt.title('Histogram of Student Absences')
plt.xlabel('Number of Days Absent')
plt.ylabel('Frequency')
plt.xticks(range(min(absences), max(absences) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
