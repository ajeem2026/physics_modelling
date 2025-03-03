#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:20:40 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------
# Simulation Parameters
# ---------------------
L = 50                  # Lattice size (L x L)
T = 2.8                 # Temperature (in units where k_B = 1)
J = 1.0                 # Coupling constant (ferromagnetic if J > 0)
H = 0.0                 # External magnetic field; set H=0.0 for no external field,
                        # or choose a nonzero value (e.g., H=0.5) for with external field.
n_steps = 10000         # Total number of Monte Carlo sweeps
sweeps_per_frame = 5    # Number of sweeps to perform between animation frames

# ---------------------
# Initialize Lattice
# ---------------------
# Spins: +1 or -1 chosen at random
spins = np.random.choice([-1, 1], size=(L, L))

# Precompute the Boltzmann factors for possible energy changes:
# For each site, the energy change when flipping is:
#   ΔE = 2 * s * (J * sum(neighbor_spins) + H)
# The sum over neighbors can be -4, -2, 0, 2, or 4 for a square lattice.
# We precompute exp(-ΔE/T) for ΔE = 2 * s * (J * n + H) where n = -4,-2,0,2,4.
possible_dE = np.array([-8*J - 2*H, -4*J - 2*H, -2*H, 4*J - 2*H, 8*J - 2*H])
boltzmann = {dE: np.exp(-dE/T) for dE in possible_dE}

# ---------------------
# Helper Functions
# ---------------------
def metropolis_sweep(spins):
    """
    Perform one Monte Carlo sweep (L*L flip attempts) using the Metropolis algorithm.
    """
    for i in range(L):
        for j in range(L):
            # Pick a random site (i, j)
            a = np.random.randint(0, L)
            b = np.random.randint(0, L)
            s = spins[a, b]
            # Periodic boundary conditions: sum of nearest neighbors
            # Neighbors: up, down, left, right
            nb = spins[(a+1)%L, b] + spins[(a-1)%L, b] + spins[a, (b+1)%L] + spins[a, (b-1)%L]
            # Energy change if spin (a, b) is flipped:
            dE = 2 * s * (J * nb + H)
            # Metropolis acceptance:
            if dE <= 0 or np.random.rand() < np.exp(-dE/T):
                spins[a, b] = -s
    return spins

# ---------------------
# Animation Setup
# ---------------------
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(spins, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_title('2D Ising Model (H = {})'.format(H))
ax.set_xticks([])
ax.set_yticks([])

def init():
    im.set_data(spins)
    return (im,)

def update(frame):
    global spins
    # Perform several sweeps between animation frames for smoother evolution.
    for _ in range(sweeps_per_frame):
        spins = metropolis_sweep(spins)
    im.set_data(spins)
    return (im,)

ani = FuncAnimation(fig, update, frames=range(n_steps//sweeps_per_frame),
                    init_func=init, interval=50, blit=True)

plt.show()
