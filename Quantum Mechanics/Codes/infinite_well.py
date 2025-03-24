#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infinite Potential Well: Energy Levels, Eigenfunctions, and State Animations

This script simulates a particle in an infinite potential well defined by:
    V(x) = 0 for 0 <= x <= L and V(x) = ∞ outside.
We perform the following steps:
  1. Discretize the spatial domain and set the potential to 0 inside the well.
  2. Construct the Hamiltonian using finite differences.
  3. Solve for the energy eigenvalues and eigenfunctions.
  4. Plot the first four eigenfunctions (shifted vertically by their energy eigenvalues).
  5. Animate the time evolution of:
     (a) The ground state (n = 0)
     (b) The first excited state (n = 1)

Note:
For any stationary eigenstate, time evolution introduces a phase factor:
    ψ(x, t) = ψ(x, 0) * exp(-i E t),
which does not change the probability density but causes the real part of the wave function to oscillate.

@author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# -------------------------
# Section 1: Discretize the Space and Define the Potential
# -------------------------
L = 1.0                     # Length of the well (0 <= x <= L)
N = 500                     # Number of grid points
x = np.linspace(0, L, N)    # Create an array of x values from 0 to L
dx = x[1] - x[0]            # Grid spacing

# For an infinite potential well, the potential is zero inside the well.
V = np.zeros(N)  # V(x) = 0 for 0 <= x <= L

# -------------------------
# Section 2: Construct the Hamiltonian using Finite Differences
# -------------------------
# The kinetic energy operator is given by: T = -0.5 * d²/dx².
# We approximate the second derivative using central differences.
T = -0.5 * (np.diag(np.ones(N-1), -1) 
            - 2 * np.diag(np.ones(N), 0) 
            + np.diag(np.ones(N-1), 1)) / dx**2

# Total Hamiltonian: H = T + V (with V added as a diagonal matrix).
H = T + np.diag(V)

# -------------------------
# Section 3: Solve the Eigenvalue Problem
# -------------------------
# Use the 'eigh' function which is optimized for symmetric (Hermitian) matrices.
eigvals, eigvecs = eigh(H)

# -------------------------
# Section 4: Plot the First Four Eigenfunctions and Their Energy Levels
# -------------------------
plt.figure(figsize=(10, 6))
for n in range(4):
    # Shift each eigenfunction vertically by its corresponding energy eigenvalue.
    plt.plot(x, eigvecs[:, n] + eigvals[n], label=f'n = {n}, E = {eigvals[n]:.2f}')
plt.xlabel('x')
plt.ylabel('Energy + Wavefunction')
plt.title('Infinite Well: Eigenfunctions and Energy Levels')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Section 5: Animate the Ground State (n = 0)
# -------------------------
# Select the ground state (lowest energy state, index 0).
psi_ground = eigvecs[:, 0]
E_ground = eigvals[0]

# Normalize the ground state: ensure that the integral of |ψ(x)|² over x equals 1.
psi_ground = psi_ground / np.sqrt(np.sum(np.abs(psi_ground)**2) * dx)

# Set up the figure for the ground state animation.
fig_ground, ax_ground = plt.subplots(figsize=(8, 4))
line_ground, = ax_ground.plot(x, np.real(psi_ground), 'b-', lw=2)
ax_ground.set_xlim(0, L)
ax_ground.set_ylim(-1.5 * np.max(np.abs(psi_ground)), 1.5 * np.max(np.abs(psi_ground)))
ax_ground.set_xlabel('x')
ax_ground.set_ylabel('Re(ψ(x,t))')
ax_ground.set_title(f'Infinite Well: Ground State (n=0), E = {E_ground:.2f}')

def update_ground(frame):
    t = frame * 0.02  # Define the current time (adjust time step as needed)
    # Apply time evolution: ψ(x, t) = ψ(x, 0) * exp(-i E t)
    psi_t = psi_ground * np.exp(-1j * E_ground * t)
    # Update the plot with the real part of the time-evolved wave function.
    line_ground.set_ydata(np.real(psi_t))
    return line_ground,

ani_ground = FuncAnimation(fig_ground, update_ground, frames=200, interval=50, blit=True)
plt.show()

# -------------------------
# Section 6: Animate the First Excited State (n = 1)
# -------------------------
# Select the first excited state (index 1).
psi_excited = eigvecs[:, 1]
E_excited = eigvals[1]

# Normalize the excited state.
psi_excited = psi_excited / np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)

# Set up the figure for the first excited state animation.
fig_excited, ax_excited = plt.subplots(figsize=(8, 4))
line_excited, = ax_excited.plot(x, np.real(psi_excited), 'r-', lw=2)
ax_excited.set_xlim(0, L)
ax_excited.set_ylim(-1.5 * np.max(np.abs(psi_excited)), 1.5 * np.max(np.abs(psi_excited)))
ax_excited.set_xlabel('x')
ax_excited.set_ylabel('Re(ψ(x,t))')
ax_excited.set_title(f'Infinite Well: First Excited State (n=1), E = {E_excited:.2f}')

def update_excited(frame):
    t = frame * 0.02  # Define the current time for this frame.
    # Time evolution: ψ(x, t) = ψ(x, 0) * exp(-i E t)
    psi_t = psi_excited * np.exp(-1j * E_excited * t)
    # Update the plot with the real part of the time-evolved excited state.
    line_excited.set_ydata(np.real(psi_t))
    return line_excited,

ani_excited = FuncAnimation(fig_excited, update_excited, frames=200, interval=50, blit=True)
plt.show()
