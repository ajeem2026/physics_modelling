#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:12:03 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_steps = 1000            # total number of steps
step_size = 1             # step size (can be negative or positive)
n_walks = 1000            # number of independent random walks to simulate

# Generate random walks
# Each walk consists of n_steps steps, and we simulate n_walks such walks
steps = np.random.choice([-step_size, step_size], size=(n_walks, n_steps))
walks = np.cumsum(steps, axis=1)  # Cumulative sum to generate the random walks

# Calculate the Mean Squared Displacement (MSD) as a function of time
msd = np.mean(walks**2, axis=0)  # MSD = <x^2>, averaged over all walks

# Time array (steps)
time = np.arange(1, n_steps + 1)

# Plot MSD versus time on a log-log scale
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(time, msd, 'bo-', lw=2, label='MSD (Simulated)')
ax.set_xlabel('Time (steps)', fontsize=14)
ax.set_ylabel('Mean Squared Displacement (MSD)', fontsize=14)
ax.set_title('MSD vs Time (Log-Log Scale)', fontsize=16)
ax.grid(True, which="both", ls="--")

# Overlay the theoretical MSD (scales linearly with time: MSD = t * step_size^2)
theoretical_msd = time * (step_size ** 2)
ax.loglog(time, theoretical_msd, 'r--', lw=2, label='Theoretical MSD (Linear Scaling)')

# Add a label to show the MSD calculation
msd_label = r'$\text{MSD} = t \cdot \text{step\_size}^2$'
ax.text(0.02, 0.95, msd_label, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add a legend
ax.legend(fontsize=12)

# Display the plot
plt.show()