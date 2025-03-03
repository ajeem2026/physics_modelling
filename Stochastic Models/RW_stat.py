#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:06:13 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Simulation Parameters
# -----------------------------
n_steps = 1000       # Number of steps per random walk
n_trials = 1000      # Number of independent realizations
step_size = 1        # Step size (magnitude)
bias_unbiased = 0.0  # Bias parameter for unbiased random walk
bias_biased = 0.2    # Bias parameter for biased random walk (range: -0.5 to +0.5)

# A helper function to simulate 1D random walks with a given bias.
def simulate_random_walks(n_trials, n_steps, step_size, bias):
    """
    Simulate a set of 1D random walks.
    
    For each step:
      - The probability to move to the right (+1) is 0.5 + bias.
      - The probability to move to the left (-1) is 0.5 - bias.
      
    Returns:
      positions: a (n_trials x (n_steps+1)) array of positions (starting at 0).
    """
    # Initialize an array for positions: each row is one walk; first column is 0.
    positions = np.zeros((n_trials, n_steps + 1))
    # Define the step options and probabilities.
    steps_options = np.array([step_size, -step_size])
    p_right = 0.5 + bias
    p_left  = 0.5 - bias
    probabilities = [p_right, p_left]
    
    # Run simulations
    for i in range(n_trials):
        # Random steps for this trial:
        steps = np.random.choice(steps_options, size=n_steps, p=probabilities)
        positions[i, 1:] = np.cumsum(steps)
    return positions

# -----------------------------
# Simulation: Unbiased and Biased Cases
# -----------------------------
positions_unbiased = simulate_random_walks(n_trials, n_steps, step_size, bias_unbiased)
positions_biased   = simulate_random_walks(n_trials, n_steps, step_size, bias_biased)

# -----------------------------
# Compute Mean Squared Displacement (MSD)
# -----------------------------
def compute_msd(positions):
    """
    Compute the mean squared displacement (MSD) over time.
    
    positions: array of shape (n_trials, n_steps+1)
    
    Returns:
      msd: 1D array of length n_steps+1
    """
    return np.mean(positions**2, axis=0)

msd_unbiased = compute_msd(positions_unbiased)
msd_biased   = compute_msd(positions_biased)
time = np.arange(n_steps + 1)

# -----------------------------
# Plot MSD on a Log-Log Scale
# -----------------------------
plt.figure(figsize=(8,6))
plt.loglog(time, msd_unbiased, label='Unbiased (bias=0)')
plt.loglog(time, msd_biased, label='Biased (bias=0.2)', linestyle='--')
plt.xlabel('Time (steps)')
plt.ylabel('Mean Squared Displacement')
plt.title('MSD vs Time (Log-Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

# -----------------------------
# Compute and Plot Probability Distribution of Displacement
# -----------------------------
def plot_distribution(positions, label, color):
    """
    Plot the empirical probability distribution of the displacement at final time,
    and overlay the corresponding Gaussian distribution.
    """
    final_positions = positions[:, -1]
    
    # Histogram of the final displacements (density normalized)
    counts, bins, _ = plt.hist(final_positions, bins=50, density=True, alpha=0.6,
                               label=f'Empirical: {label}', color=color)
    
    # Calculate parameters for the Gaussian:
    # For an unbiased walk, the expected mean is zero; for biased, the mean is nonzero.
    mu = np.mean(final_positions)
    sigma = np.std(final_positions)
    
    # Generate x-values and compute Gaussian probability density
    x_vals = np.linspace(bins[0], bins[-1], 200)
    gaussian = norm.pdf(x_vals, loc=mu, scale=sigma)
    
    plt.plot(x_vals, gaussian, color=color, linewidth=2,
             label=f'Gaussian fit: {label}\n$\mu$={mu:.2f}, $\sigma$={sigma:.2f}')

plt.figure(figsize=(8,6))
plot_distribution(positions_unbiased, label='Unbiased', color='blue')
plot_distribution(positions_biased, label='Biased', color='red')
plt.xlabel('Displacement')
plt.ylabel('Probability Density')
plt.title('Probability Distribution of Displacement at Final Time')
plt.legend()
plt.grid(True, ls="--", lw=0.5)
plt.show()
