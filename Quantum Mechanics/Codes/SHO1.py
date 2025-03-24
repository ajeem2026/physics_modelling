#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:49:40 2025

This script demonstrates the simple harmonic oscillator (SHO) by:
1. Plotting the first four eigenfunctions (eigenvectors) with their corresponding energy levels.
2. Animating the time evolution (phase evolution) of the ground state.
3. Animating the time evolution of the first excited state.

Note:
For any stationary eigenstate, the time evolution introduces only a phase factor:
    ψ(x,t) = ψ(x,0) * exp(-i E t)
which does not affect the probability density. Here, we animate the real part to illustrate the evolution.

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# -------------------------
# Section 1: Setup & Eigenvalue Problem
# -------------------------
# Step 1: Discretize the space
N = 500                   # Number of grid points
x_min, x_max = -10, 10    # Spatial range from -10 to 10
x = np.linspace(x_min, x_max, N)  # Array of x values
dx = x[1] - x[0]          # Grid spacing

# Step 2: Define the SHO potential
# Using atomic units: m = 1, ω = 1 so that V(x) = 0.5 * x^2
V = 0.5 * x**2

# Step 3: Construct the Hamiltonian using finite differences
# Kinetic energy operator: T = -0.5 * d²/dx² (approximated using central differences)
T = -0.5 * (
    np.diag(np.ones(N - 1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N - 1), 1)
) / dx**2
# Total Hamiltonian: H = T + V (with V as a diagonal matrix)
H = T + np.diag(V)

# Step 4: Solve the eigenvalue problem to obtain energy levels and eigenfunctions
eigvals, eigvecs = eigh(H)

# -------------------------
# Section 2: Plot Energy Levels and Eigenfunctions
# -------------------------
plt.figure(figsize=(10, 6))
# Plot the first four eigenfunctions, each shifted vertically by its eigenvalue
for n in range(4):
    plt.plot(x, eigvecs[:, n] + eigvals[n], label=f'n = {n}, E = {eigvals[n]:.2f}')
plt.xlabel('x')
plt.ylabel('Energy + Wavefunction')
plt.title('Eigenfunctions of the Simple Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Section 3: Animate Ground State Time Evolution
# -------------------------
# The ground state is the lowest eigenvalue state (n = 0)
psi_ground = eigvecs[:, 0]
E_ground = eigvals[0]

# Normalize the ground state so that ∫|ψ(x)|^2 dx = 1
psi_ground = psi_ground / np.sqrt(np.sum(np.abs(psi_ground)**2) * dx)

# Set up the figure for the ground state animation
fig_ground, ax_ground = plt.subplots(figsize=(8, 4))
line_ground, = ax_ground.plot(x, np.real(psi_ground), 'b-', lw=2)
ax_ground.set_xlim(x_min, x_max)
ax_ground.set_ylim(-1.5 * np.max(np.abs(psi_ground)), 1.5 * np.max(np.abs(psi_ground)))
ax_ground.set_xlabel('x')
ax_ground.set_ylabel('Re(ψ(x,t))')
ax_ground.set_title(f'SHO Ground State: Re(ψ(x,t)), E = {E_ground:.2f}')

def update_ground(frame):
    t = frame * 0.05  # Time for the current frame
    # Time evolution: ψ(x,t) = ψ(x,0) * exp(-i * E * t)
    psi_t = psi_ground * np.exp(-1j * E_ground * t)
    line_ground.set_ydata(np.real(psi_t))
    return line_ground,

ani_ground = FuncAnimation(fig_ground, update_ground, frames=200, interval=50, blit=True)

# Display the ground state animation
plt.show()

# -------------------------
# Section 4: Animate First Excited State Time Evolution
# -------------------------
# The first excited state corresponds to n = 1 (index 1, since index 0 is ground state)
psi_excited = eigvecs[:, 1]
E_excited = eigvals[1]

# Normalize the first excited state
psi_excited = psi_excited / np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)

# Set up the figure for the first excited state animation
fig_excited, ax_excited = plt.subplots(figsize=(8, 4))
line_excited, = ax_excited.plot(x, np.real(psi_excited), 'r-', lw=2)
ax_excited.set_xlim(x_min, x_max)
ax_excited.set_ylim(-1.5 * np.max(np.abs(psi_excited)), 1.5 * np.max(np.abs(psi_excited)))
ax_excited.set_xlabel('x')
ax_excited.set_ylabel('Re(ψ(x,t))')
ax_excited.set_title(f'SHO First Excited State: Re(ψ(x,t)), E = {E_excited:.2f}')

def update_excited(frame):
    t = frame * 0.05  # Time for the current frame
    # Time evolution of the excited state: ψ(x,t) = ψ(x,0) * exp(-i * E * t)
    psi_t = psi_excited * np.exp(-1j * E_excited * t)
    line_excited.set_ydata(np.real(psi_t))
    return line_excited,

ani_excited = FuncAnimation(fig_excited, update_excited, frames=200, interval=50, blit=True)

# Display the excited state animation
plt.show()
