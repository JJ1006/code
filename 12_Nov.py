import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the provided CSV file
data_path = 'covid_data_prog_assign_3-1.csv'
covid_data = pd.read_csv(data_path)

# Separate data by red and blue states
red_states = covid_data[covid_data['red_blue'] == 'red']
blue_states = covid_data[covid_data['red_blue'] == 'blue']

# ------------------------------
# Question 1: Part A
# Calculate observed mean differences in deaths per 100k and cases per 100k
# ------------------------------
observed_diff_deaths = red_states['deaths_avg_per_100k_mean'].mean() - blue_states['deaths_avg_per_100k_mean'].mean()
observed_diff_cases = red_states['cases_avg_per_100k_mean'].mean() - blue_states['cases_avg_per_100k_mean'].mean()

# Define function to generate bootstrap samples and calculate mean difference
def bootstrap_mean_diff(data_red, data_blue, num_samples=1000):
    bootstrapped_diffs = []
    for _ in range(num_samples):
        sample_red = data_red.sample(frac=1, replace=True)
        sample_blue = data_blue.sample(frac=1, replace=True)
        diff = sample_red.mean() - sample_blue.mean()
        bootstrapped_diffs.append(diff)
    return np.array(bootstrapped_diffs)

# ------------------------------
# Question 1: Part B
# Generate 95% Bootstrap Confidence Interval for mean difference in deaths per 100k
# ------------------------------
boot_diff_deaths = bootstrap_mean_diff(red_states['deaths_avg_per_100k_mean'], blue_states['deaths_avg_per_100k_mean'])
ci_deaths = np.percentile(boot_diff_deaths, [2.5, 97.5])
print("95% Confidence Interval for Deaths per 100k mean difference:", ci_deaths)

# ------------------------------
# Question 3: Part A
# Generate 95% Bootstrap Confidence Interval for mean difference in cases per 100k
# ------------------------------
boot_diff_cases = bootstrap_mean_diff(red_states['cases_avg_per_100k_mean'], blue_states['cases_avg_per_100k_mean'])
ci_cases = np.percentile(boot_diff_cases, [2.5, 97.5])
print("95% Confidence Interval for Cases per 100k mean difference:", ci_cases)

# ------------------------------
# Question 3: Part B
# Plot histograms with confidence intervals for both deaths and cases
# ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=100)

# Styling variables
color_deaths = 'skyblue'
color_cases = 'lightcoral'
line_color_ci = 'red'
line_color_obs = 'green'

# Histogram for deaths per 100k mean difference
axs[0].hist(boot_diff_deaths, bins=30, color=color_deaths, edgecolor='black')
axs[0].axvline(ci_deaths[0], color=line_color_ci, linestyle='dotted', linewidth=2, label=f'Lower CI: {ci_deaths[0]:.2f}')
axs[0].axvline(ci_deaths[1], color=line_color_ci, linestyle='dotted', linewidth=2, label=f'Upper CI: {ci_deaths[1]:.2f}')
axs[0].axvline(observed_diff_deaths, color=line_color_obs, linestyle='solid', linewidth=2, label=f'Observed Diff: {observed_diff_deaths:.2f}')
axs[0].set_title("Bootstrap Distribution for Deaths Avg per 100k Mean Difference")
axs[0].set_xlabel("Mean Difference (Deaths per 100k)")
axs[0].set_ylabel("Frequency")
axs[0].legend(loc="upper left")

# Histogram for cases per 100k mean difference
axs[1].hist(boot_diff_cases, bins=30, color=color_cases, edgecolor='black')
axs[1].axvline(ci_cases[0], color=line_color_ci, linestyle='dotted', linewidth=2, label=f'Lower CI: {ci_cases[0]:.2f}')
axs[1].axvline(ci_cases[1], color=line_color_ci, linestyle='dotted', linewidth=2, label=f'Upper CI: {ci_cases[1]:.2f}')
axs[1].axvline(observed_diff_cases, color=line_color_obs, linestyle='solid', linewidth=2, label=f'Observed Diff: {observed_diff_cases:.2f}')
axs[1].set_title("Bootstrap Distribution for Cases Avg per 100k Mean Difference")
axs[1].set_xlabel("Mean Difference (Cases per 100k)")
axs[1].set_ylabel("Frequency")
axs[1].legend(loc="upper left")

plt.tight_layout()
plt.show()

# Final output for both confidence intervals
print("\nSummary of Results:")
print("95% CI for Deaths per 100k mean difference:", ci_deaths)
print("95% CI for Cases per 100k mean difference:", ci_cases)
