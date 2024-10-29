import numpy as np

# Constants
coin_values = np.array([1, 5, 10, 25, 50, 100])  # Possible coin values (face values)
observed_mean = 10.3828  # Given observed mean we want to match

def coin_weights(alpha, coin_obs_mean=observed_mean, coin_value=coin_values, print_probs=False):
    """
    Calculate the probabilities and mean difference for a given alpha.
    
    Parameters:
    - alpha (float): The Lagrange multiplier to be optimized.
    - coin_obs_mean (float): The observed mean to match.
    - coin_value (array): Array of coin face values.
    - print_probs (bool): If True, returns detailed probability info.
    
    Returns:
    - float or dict: Mean difference if print_probs=False, or detailed info otherwise.
    """
    # Clip values to prevent overflow in exp function
    exp_alpha_x = np.exp(np.clip(alpha * coin_value, -700, 700))  # Limits to avoid overflow
    prob_coin = exp_alpha_x / np.sum(exp_alpha_x)  # Normalized probabilities
    estimated_mean = np.sum(coin_value * prob_coin)  # Mean based on these probabilities
    mean_diff = abs(estimated_mean - coin_obs_mean)  # Absolute difference from observed mean
    
    # Return loss (mean difference) or detailed info based on flag
    if not print_probs:
        return mean_diff  # For optimization purposes
    else:
        return {
            'prob_coin': prob_coin,         # Probabilities for each coin
            'estimated_mean': estimated_mean,
            'mean_diff': mean_diff
        }

def bisection_method(func, interval, tol=0.00001, max_iter=1000):
    """
    Bisection method to find the root of a function within a given tolerance.
    
    Parameters:
    - func (callable): The function whose root is sought.
    - interval (tuple): Interval (a, b) within which the root is located.
    - tol (float): Tolerance level for stopping condition.
    - max_iter (int): Maximum number of iterations allowed.
    
    Returns:
    - float: The estimated root value.
    """
    a, b = interval
    iter_count = 0
    
    # Iterate until interval is smaller than tolerance or max iterations are reached
    while abs(b - a) > tol and iter_count < max_iter:
        mid = (a + b) / 2.0  # Midpoint of the current interval
        f_mid = func(mid)    # Evaluate the function at the midpoint
        
        # Determine the side on which to continue the search
        if func(a) * f_mid < 0:
            b = mid
        else:
            a = mid
        iter_count += 1
    
    return (a + b) / 2.0  # Final midpoint as root estimate

# Step 1: Find the optimal alpha using the bisection method with limited range
optimal_alpha = bisection_method(lambda alpha: coin_weights(alpha), interval=(-1, 1), tol=0.00001)

# Step 2: Calculate the optimal probabilities and mean with the found alpha
results = coin_weights(optimal_alpha, print_probs=True)
optimal_probabilities = results['prob_coin']
estimated_mean = results['estimated_mean']
loss_value = results['mean_diff']

# Output the results
print(f"Optimal alpha: {optimal_alpha:.5f}")
print("Optimal probabilities for each coin type:")
for i, prob in enumerate(optimal_probabilities, start=1):
    print(f"  Coin {i} (value ${coin_values[i-1]}): Probability = {prob:.5f}")
print(f"Estimated mean: {estimated_mean:.5f}")
print(f"Loss function value (difference in mean): {loss_value:.5f}")
