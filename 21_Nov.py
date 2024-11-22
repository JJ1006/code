import numpy as np

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

# Define the "wiggly function" as in the R file
def wiggly_function(x):
    return -np.exp(x[0]**2 / 100) * np.sin(13 * x[0] - x[0]**4)**5 * np.sin(1 - 3 * x[0]**2)**2

# Parameters matching the R implementation
x_initial = [0]  # Starting value for x
x_bounds = (-2, 2)  # Bounds for x
temp_initial = 2  # Initial temperature
temp_lb = 0.0001  # Lower bound for temperature
cooling_rate = 0.0001  # Cooling rate

# Run simulated annealing
print("Running Simulated Annealing...\n")
x_optimal, y_min, solution_path = simulated_annealing(
    wiggly_function,
    x_initial,
    x_bounds=x_bounds,
    temp=temp_initial,
    temp_lb=temp_lb,
    rate=cooling_rate
)

# Display results
print("\n=== Final Results ===")
print(f"Optimal x: {x_optimal[0]:.6f}")
print(f"Minimum f(x): {y_min:.6f}")
print("\nSolution Path (last 5 steps):")
for step in solution_path[-5:]:
    print(f"x: {step[0]:.6f}, f(x): {step[1]:.6f}")
