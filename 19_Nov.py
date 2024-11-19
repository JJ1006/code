import numpy as np

# Simulated Annealing Algorithm
def simulated_annealing(f, x_init, x_bounds=(-2, 2), temp=2, temp_lb=0.0001, rate=0.0001):
    """
    Simulated Annealing algorithm to minimize a given function.
    
    Args:
        f: The function to minimize.
        x_init: Initial guess for the variable(s).
        x_bounds: Tuple (lower_bound, upper_bound) for x.
        temp: Initial temperature.
        temp_lb: Lower bound for temperature.
        rate: Cooling rate.
        
    Returns:
        x_best: The variable(s) that minimize the function.
        y_best: The minimum function value.
    """
    step = 1 - rate  # Temperature decay factor
    x_curr = np.array(x_init, dtype=float)
    x_best = np.copy(x_curr)
    y_curr = y_best = f(x_curr)
    
    while temp > temp_lb:
        # Generate a new candidate solution
        x_next = x_curr + np.random.normal(0, temp, size=x_curr.shape)
        
        # Keep x_next within bounds
        x_next = np.clip(x_next, x_bounds[0], x_bounds[1])
        
        # Evaluate the new candidate
        y_next = f(x_next)
        
        # Calculate the acceptance probability
        accept_prob = np.exp(-(y_next - y_curr) / temp)
        
        # Accept or reject the candidate
        if (y_next < y_curr) or (np.random.rand() < accept_prob):
            x_curr = x_next
            y_curr = y_next
        
        # Update the best solution if the candidate is better
        if y_next < y_best:
            x_best = x_next
            y_best = y_next
        
        # Reduce the temperature
        temp *= step
    
    return x_best, y_best

# Part 1: Minimizing the "Wiggly Function"
def wiggly_function(x):
    """
    The "wiggly function" to minimize.
    """
    return -np.exp(x[0]**2 / 100) * np.sin(13 * x[0] - x[0]**4)**5 * np.sin(1 - 3 * x[0]**2)**2

# Solve Part 1
x_initial = [0]  # Initial guess for x
x_optimal, y_min = simulated_annealing(wiggly_function, x_initial)

# Print results for Part 1
print("\n=== Part 1: Minimizing the 'Wiggly Function' ===")
print(f"Optimal x: {x_optimal[0]:.4f}")
print(f"Minimum value of f(x): {y_min:.4f}")

# Part 2: Optimizing Probabilities for Coin Weights
def coin_weights(alpha):
    """
    Compute probabilities for given alpha and coin values.
    """
    coin_values = np.array([1, 5, 10, 25, 50, 100])
    exp_values = np.exp(-alpha * coin_values)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def loss_function(alpha):
    """
    Compute the loss function as the absolute difference between the observed and calculated mean.
    """
    observed_mean = 10.3828  # Given observed mean
    coin_values = np.array([1, 5, 10, 25, 50, 100])
    probabilities = coin_weights(alpha)
    calculated_mean = np.sum(probabilities * coin_values)
    return abs(calculated_mean - observed_mean)

# Solve Part 2 using Simulated Annealing
def optimize_alpha():
    """
    Use Simulated Annealing to find the optimal alpha that minimizes the loss function.
    """
    alpha_bounds = (0.0001, 5)  # Reasonable bounds for alpha
    alpha_initial = [1.0]  # Initial guess for alpha
    
    # Run simulated annealing
    alpha_optimal, loss_min = simulated_annealing(
        lambda alpha: loss_function(alpha),
        alpha_initial,
        x_bounds=alpha_bounds,
        temp=2,
        temp_lb=0.0001,
        rate=0.001
    )
    
    # Calculate final probabilities and mean
    optimal_alpha = alpha_optimal[0]
    final_probabilities = coin_weights(optimal_alpha)
    final_mean = np.sum(final_probabilities * np.array([1, 5, 10, 25, 50, 100]))
    observed_mean = 10.3828
    final_loss = abs(final_mean - observed_mean)
    
    return optimal_alpha, final_probabilities, final_mean, final_loss

# Solve Part 2
optimal_alpha, final_probabilities, final_mean, final_loss = optimize_alpha()

# Print results for Part 2
print("\n=== Part 2: Optimizing Probabilities for Coin Weights ===")
print(f"Optimal alpha: {optimal_alpha:.4f}")
print("Estimated probabilities for each coin value:")
coin_values = [1, 5, 10, 25, 50, 100]
for coin, prob in zip(coin_values, final_probabilities):
    print(f"  Coin value {coin}: {prob:.4f}")
print(f"Estimated mean: {final_mean:.4f}")
print(f"Loss (difference from observed mean): {final_loss:.4f}")
