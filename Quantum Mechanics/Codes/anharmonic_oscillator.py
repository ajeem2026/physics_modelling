#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4(a): Anharmonic Oscillator – Energy Levels, Eigenfunctions, and Excited State Animation

This script simulates an anharmonic oscillator with the potential:
    V(x) = 0.5 * x^2 + λ * x^4,
where λ is a small anharmonicity parameter.
We:
  1. Discretize the spatial domain.
  2. Define the anharmonic potential.
  3. Construct the Hamiltonian using finite differences.
  4. Solve for the energy eigenvalues and eigenfunctions.
  5. Plot the first four eigenfunctions (each individually scaled so that its maximum amplitude is fixed, then shifted by its energy eigenvalue)
     to clearly display the oscillatory features (minima and maxima).
  6. Select and animate the time evolution of the first excited state (n = 1).

The time evolution is given by: 
    ψ(x,t) = ψ(x,0) exp(-i E t)
which, while leaving the probability density unchanged, causes the real part of the wave function to oscillate.

@author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# -------------------------
# Step 1: Discretize the Space
# -------------------------
lambda_val = 0.05          # Anharmonicity parameter (small deviation from harmonic oscillator)
N = 500                    # Number of spatial grid points
x_min, x_max = -5, 5       # Spatial domain covering the region of interest
x = np.linspace(x_min, x_max, N)  # Array of x values
dx = x[1] - x[0]           # Grid spacing

# -------------------------
# Step 2: Define the Anharmonic Potential
# -------------------------
# The anharmonic potential is defined as:
#   V(x) = 0.5 * x^2 + λ * x^4
V_anharm = 0.5 * x**2 + lambda_val * x**4

# -------------------------
# Step 3: Construct the Hamiltonian using Finite Differences
# -------------------------
# Use central differences to approximate the second derivative (kinetic energy term)
T = -0.5 * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)) / dx**2
# The total Hamiltonian is the sum of kinetic (T) and potential (V) energy operators.
H_anharm = T + np.diag(V_anharm)

# -------------------------
# Step 4: Solve the Eigenvalue Problem
# -------------------------
# Compute the energy eigenvalues and eigenfunctions using the symmetric eigenvalue solver.
eigvals, eigvecs = eigh(H_anharm)

# -------------------------
# Step 5: Plot Eigenfunctions and Energy Levels with Improved Scaling
# -------------------------
plt.figure(figsize=(10, 6))
for n in range(4):
    # Scale each eigenfunction so its maximum absolute value becomes 0.5.
    scale_n = 0.5 / np.max(np.abs(eigvecs[:, n]))
    # Plot the scaled eigenfunction, shifted vertically by its corresponding energy eigenvalue.
    plt.plot(x, scale_n * eigvecs[:, n] + eigvals[n], label=f'n = {n}, E = {eigvals[n]:.2f}')
plt.xlabel('x')
plt.ylabel('Energy + Scaled Wavefunction')
plt.title('Anharmonic Oscillator: Eigenfunctions and Energy Levels (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Step 6: Select and Normalize the First Excited State (n = 1)
# -------------------------
n = 1  # Choose the first excited state (n = 1)
psi_excited = eigvecs[:, n]  # Extract the eigenfunction corresponding to n = 1
E_excited = eigvals[n]       # Extract the corresponding energy eigenvalue

# Normalize the selected eigenstate so that ∫|ψ(x)|² dx = 1.
psi_excited = psi_excited / np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)

# -------------------------
# Step 7: Animate the Time Evolution of the First Excited State
# -------------------------
# Even though a stationary eigenstate's probability density is time-independent,
# its real part oscillates in time due to the phase factor exp(-i E t).
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, np.real(psi_excited), 'r-', lw=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.5 * np.max(np.abs(psi_excited)), 1.5 * np.max(np.abs(psi_excited)))
ax.set_xlabel('x')
ax.set_ylabel('Re(ψ(x,t))')
ax.set_title(f'Anharmonic Oscillator: First Excited State (n=1), E = {E_excited:.2f}')

def update(frame):
    t = frame * 0.05  # Time for the current frame
    # Time evolution: apply phase factor exp(-i E t) to the initial eigenstate.
    psi_t = psi_excited * np.exp(-1j * E_excited * t)
    # Update the plot with the real part of the time-evolved wave function.
    line.set_ydata(np.real(psi_t))
    return line,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
