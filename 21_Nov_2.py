import numpy as np
import matplotlib.pyplot as plt

# Part 1: Simulated Annealing for the "Wiggly Function"
def simulated_annealing(f, x_init, x_bounds=(-2, 2), temp=2, temp_lb=0.0001, rate=0.0001):
    # Initialize variables
    step = 1 - rate
    x_curr = np.array(x_init, dtype=float)
    x_best = np.copy(x_curr)
    y_curr = y_best = f(x_curr)
    solution_path = [(x_curr[0], y_curr)]  # Track solutions

    print(f"Initial x: {x_curr[0]:.6f}, f(x): {y_curr:.6f}")
    
    # Annealing process
    while temp > temp_lb:
        # Generate a new candidate solution
        x_next = x_curr + np.random.normal(0, temp, size=x_curr.shape)
        x_next = np.clip(x_next, x_bounds[0], x_bounds[1])  # Ensure within bounds
        
        # Evaluate the candidate solution
        y_next = f(x_next)
        
        # Metropolis acceptance criterion
        accept_prob = np.exp(-(y_next - y_curr) / temp)
        if (y_next < y_curr) or (np.random.rand() < accept_prob):
            x_curr = x_next
            y_curr = y_next
        
        # Update the best solution found
        if y_next < y_best:
            x_best = x_next
            y_best = y_next
            solution_path.append((x_best[0], y_best))
            print(f"Updated Best -> x: {x_best[0]:.6f}, f(x): {y_best:.6f}")
        
        # Reduce the temperature
        temp *= step

    return x_best, y_best, solution_path

def wiggly_function(x):
    """
    Wiggly function defined in the R file.
    """
    return -np.exp(x[0]**2 / 100) * np.sin(13 * x[0] - x[0]**4)**5 * np.sin(1 - 3 * x[0]**2)**2

def plot_wiggly_function(solution_path):
    """
    Plot the wiggly function and overlay the solution path.
    """
    x_vals = np.linspace(-2, 2, 10000)
    y_vals = [wiggly_function([x]) for x in x_vals]

    # Plot the wiggly function
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="Wiggly Function", color="blue")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Wiggly Function with Solution Path")
    plt.ylim(-1, 1)

    # Overlay the solution path
    path_x = [step[0] for step in solution_path]
    path_y = [step[1] for step in solution_path]
    plt.scatter(path_x, path_y, color="red", label="Solution Path", s=10)
    plt.legend()
    plt.grid()
    plt.show()

# Part 2: Coin Jar Problem
def coin_weights(alpha):
    """
    Compute probabilities for given alpha and coin values.
    """
    coin_values = np.array([1, 5, 10, 25, 50, 100])
    exp_values = np.exp(-alpha * coin_values)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def loss_function(alpha):
    observed_mean = 10.3828  # Given observed mean
    coin_values = np.array([1, 5, 10, 25, 50, 100])
    probabilities = coin_weights(alpha)
    calculated_mean = np.sum(probabilities * coin_values)
    return abs(calculated_mean - observed_mean)

def optimize_alpha():

    alpha_bounds = (0.0001, 5)  # Reasonable bounds for alpha
    alpha_initial = [1.0]  # Initial guess for alpha
    
    # Run simulated annealing
    alpha_optimal, loss_min, _ = simulated_annealing(
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
    final_loss = abs(final_mean - 10.3828)
    
    return optimal_alpha, final_probabilities, final_mean, final_loss

# Run Part 1
print("=== Part 1: Minimizing the Wiggly Function ===\n")
x_initial = [0]  # Starting value for x
x_bounds = (-2, 2)  # Bounds for x
temp_initial = 2  # Initial temperature
temp_lb = 0.0001  # Lower bound for temperature
cooling_rate = 0.0001  # Cooling rate
x_optimal, y_min, solution_path = simulated_annealing(
    wiggly_function,
    x_initial,
    x_bounds=x_bounds,
    temp=temp_initial,
    temp_lb=temp_lb,
    rate=cooling_rate
)
print(f"\nOptimal x: {x_optimal[0]:.6f}")
print(f"Minimum f(x): {y_min:.6f}")
plot_wiggly_function(solution_path)

# Run Part 2
print("\n=== Part 2: Optimizing Probabilities for Coin Weights ===\n")
optimal_alpha, final_probabilities, final_mean, final_loss = optimize_alpha()
print(f"Optimal alpha: {optimal_alpha:.6f}")
print("Estimated probabilities for each coin value:")
coin_values = [1, 5, 10, 25, 50, 100]
for coin, prob in zip(coin_values, final_probabilities):
    print(f"  Coin value {coin}: {prob:.6f}")
print(f"Estimated mean: {final_mean:.6f}")
print(f"Loss (difference from observed mean): {final_loss:.6f}")
