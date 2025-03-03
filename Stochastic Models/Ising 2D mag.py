#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:17:07 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Simulation Parameters
# -----------------------------
L = 50              # Lattice size (L x L)
T = 2.5             # Temperature (in units where k_B = 1)
J = 1.0             # Coupling constant (J > 0 for ferromagnetic interactions)
H = 0.0             # External magnetic field (set H=0.0 for no external field)
n_sweeps = 5000     # Total number of Monte Carlo sweeps to simulate
sweeps_per_frame = 10  # Number of sweeps between each animation frame

# -----------------------------
# Initialize Lattice and Observables
# -----------------------------
# Initialize spins randomly as +1 or -1
spins = np.random.choice([-1, 1], size=(L, L))
mag_history = []    # To record magnetization over time
sweep_history = []  # To record the sweep numbers

# -----------------------------
# Metropolis Sweep Function
# -----------------------------
def metropolis_sweep(spins):
    """
    Perform one Monte Carlo sweep over the lattice.
    For each spin in the lattice, attempt a flip according to the Metropolis algorithm.
    """
    for _ in range(L * L):
        # Choose a random lattice site
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        # Calculate the sum of nearest neighbors (with periodic boundaries)
        nb = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
        # Energy change if spin (i,j) is flipped:
        dE = 2 * s * (J * nb + H)
        # Accept flip if energy decreases or with probability exp(-dE/T)
        if dE <= 0 or np.random.rand() < np.exp(-dE/T):
            spins[i, j] = -s
    return spins

# -----------------------------
# Set Up Figure and Subplots
# -----------------------------
fig, (ax_lattice, ax_mag) = plt.subplots(1, 2, figsize=(12, 6))

# Lattice subplot: use imshow to display the spin configuration
im = ax_lattice.imshow(spins, cmap='coolwarm', vmin=-1, vmax=1)
ax_lattice.set_title('2D Ising Model Lattice')
ax_lattice.axis('off')

# Magnetization subplot: plot magnetization vs. Monte Carlo sweeps
ax_mag.set_title('Magnetization vs Monte Carlo Sweeps')
ax_mag.set_xlabel('Sweep Number')
ax_mag.set_ylabel('Magnetization per Spin')
mag_line, = ax_mag.plot([], [], 'b-', lw=2)
ax_mag.set_xlim(0, n_sweeps)
ax_mag.set_ylim(-1, 1)

current_sweep = 0  # Global counter for sweeps

# -----------------------------
# Animation Update Function
# -----------------------------
def update(frame):
    global spins, current_sweep
    # Perform several sweeps per animation frame
    for _ in range(sweeps_per_frame):
        spins = metropolis_sweep(spins)
        current_sweep += 1
        # Compute magnetization per spin and record it
        mag = np.mean(spins)
        mag_history.append(mag)
        sweep_history.append(current_sweep)
    
    # Update lattice image and magnetization plot
    im.set_data(spins)
    mag_line.set_data(sweep_history, mag_history)
    
    return im, mag_line

# -----------------------------
# Create Animation
# -----------------------------
ani = FuncAnimation(fig, update, frames=range(n_sweeps // sweeps_per_frame),
                    interval=50, blit=True)

plt.tight_layout()
plt.show()
