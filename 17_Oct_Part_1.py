import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Data
data = {
    'Location': ['City', 'Country', 'Total'],
    'Abroad': [5, 25, 30],
    'Home': [5, 15, 20],
    'Total': [10, 40, 50]
}

# Create DataFrame
df = pd.DataFrame(data)

def print_table(df):
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

def calculate_probability(condition, event, df):
    total = df.loc[df['Location'] == condition, 'Total'].values[0]
    count = df.loc[df['Location'] == condition, event].values[0]
    return count / total

def plot_distribution(df):
    df_plot = df.iloc[:-1]  # Exclude the 'Total' row for plotting
    df_plot.plot(x='Location', y=['Abroad', 'Home'], kind='bar', stacked=True)
    plt.title('Holiday Distribution')
    plt.xlabel('Location')
    plt.ylabel('Number of People')
    plt.legend(title='Holiday Type')
    plt.tight_layout()
    plt.show()

# Main execution
print("Holiday Distribution Table:")
print_table(df)

print("\nQuestion 1:")
prob_country_abroad = calculate_probability('Country', 'Abroad', df)
print(f"The probability that someone chosen at random from this group and known to live in the country took their holidays abroad is: {prob_country_abroad:.2%}")

print("\nQuestion 2:")
print(f"P(abroad | country) = {prob_country_abroad:.2%}")

# Visualize the distribution
plot_distribution(df)
