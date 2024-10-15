import math

def binomial_coefficient(n, k):
    return math.comb(n, k)

def binomial_probability(n, k, p):
    return binomial_coefficient(n, k) * (p ** k) * ((1 - p) ** (n - k))

def cumulative_binomial_probability(n, r, p):
    return sum(binomial_probability(n, k, p) for k in range(r + 1))

# Get user input
print("Please enter the following parameters:")

n = int(input("N (number of Bernoulli trials): "))
r = int(input("r (number of successful outcomes): "))
p = float(input("p (probability of success on a single trial, between 0 and 1): "))

# Calculate the cumulative probability
p_observed = cumulative_binomial_probability(n, r, p)

# Display the result
print(f"\nThe probability p(r <= {r}) for Binomial Distribution B({n}, {p}) is:")
print(f"p_observed = {p_observed:.6f}")

# Calculate and display individual probabilities
print("\nIndividual probabilities:")
individual_probs = []
for k in range(n + 1): 
    prob = binomial_probability(n, k, p)
    individual_probs.append(prob)
    print(f"P(X = {k}) = {prob:.6f}")

# Calculate and display the sum of individual probabilities
sum_individual_probs = sum(individual_probs)
print(f"\nSum of all individual probabilities: {sum_individual_probs:.6f}")

# Verify that the sum is close to 1 (accounting for potential floating-point imprecision)
if abs(sum_individual_probs - 1) < 1e-10:
    print("The sum of all probabilities is equal to 1, as expected.")
else:
    print("Warning: The sum of all probabilities is not exactly 1. This may be due to floating-point arithmetic limitations.")

# Print the final answer
print(f"\nFinal Answer: The probability p(r <= {r}) = {p_observed:.6f}")
