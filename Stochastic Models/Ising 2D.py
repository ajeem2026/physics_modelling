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
H = 0.5                 # External magnetic field (nonzero value)
n_steps = 10000         # Total number of Monte Carlo sweeps
sweeps_per_frame = 5    # Number of sweeps to perform between animation frames

# ---------------------
# Initialize Lattice
# ---------------------
# Spins: +1 or -1 chosen at random
spins = np.random.choice([-1, 1], size=(L, L))

# Precompute the Boltzmann factors for possible energy changes:
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
im = ax1.imshow(spins, cmap='coolwarm', vmin=-1, vmax=1)
ax1.set_title('2D Ising Model (H = {})'.format(H))
ax1.set_xticks([])
ax1.set_yticks([])

# Magnetization plot
magnetization = []  # Store magnetization per spin over time
time = []           # Store time steps
line, = ax2.plot([], [], lw=2, color='blue')
ax2.set_xlabel('Monte Carlo Sweeps')
ax2.set_ylabel('Magnetization per Spin')
ax2.set_title('Magnetization vs Time')
ax2.set_xlim(0, n_steps)
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True)

def init():
    im.set_data(spins)
    line.set_data([], [])
    return (im, line)

def update(frame):
    global spins, magnetization, time
    # Perform several sweeps between animation frames for smoother evolution.
    for _ in range(sweeps_per_frame):
        spins = metropolis_sweep(spins)
    # Calculate magnetization per spin
    m = np.mean(spins)
    magnetization.append(m)
    time.append(frame * sweeps_per_frame)
    # Update spin configuration plot
    im.set_data(spins)
    # Update magnetization plot
    line.set_data(time, magnetization)
    return (im, line)

ani = FuncAnimation(fig, update, frames=range(n_steps//sweeps_per_frame),
                    init_func=init, interval=50, blit=True)

plt.show()