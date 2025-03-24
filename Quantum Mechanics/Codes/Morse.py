#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4(b): Morse Potential – Energy Levels, Eigenfunctions, and Ground State Animation

This script simulates the Morse potential, which is used to model diatomic molecular bonds:
    V(x) = D_e * (1 - exp(-a*(x - x_e)))^2,
where D_e is the dissociation energy, a controls the width, and x_e is the equilibrium bond length.
We:
  1. Discretize the spatial domain.
  2. Define the Morse potential.
  3. Construct the Hamiltonian using finite differences.
  4. Solve for the energy eigenvalues and eigenfunctions.
  5. Plot the first four eigenfunctions (each scaled individually and shifted by its energy eigenvalue)
     to clearly show the oscillatory behavior.
  6. Animate the time evolution of the ground state (n = 0).

Time evolution: ψ(x,t) = ψ(x,0) exp(-i E t).

@author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# -------------------------
# Step 1: Discretize the Space
# -------------------------
De = 8.0                # Dissociation energy
a_morse = 0.8           # Controls the width of the potential
x_e = 0.0               # Equilibrium bond length
N = 500                 # Number of grid points
x_min, x_max = -5, 5    # Spatial domain
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# -------------------------
# Step 2: Define the Morse Potential
# -------------------------
# V(x) = D_e * (1 - exp(-a*(x - x_e)))^2
V_morse = De * (1 - np.exp(-a_morse * (x - x_e)))**2

# -------------------------
# Step 3: Construct the Hamiltonian using Finite Differences
# -------------------------
T = -0.5 * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)) / dx**2
H_morse = T + np.diag(V_morse)

# -------------------------
# Step 4: Solve the Eigenvalue Problem
# -------------------------
eigvals, eigvecs = eigh(H_morse)

# -------------------------
# Step 5: Plot Eigenfunctions and Energy Levels with Improved Scaling
# -------------------------
plt.figure(figsize=(10, 6))
for n in range(4):
    # Scale each eigenfunction so its maximum absolute value becomes 0.5.
    scale_n = 0.5 / np.max(np.abs(eigvecs[:, n]))
    plt.plot(x, scale_n * eigvecs[:, n] + eigvals[n], label=f'n = {n}, E = {eigvals[n]:.2f}')
plt.xlabel('x')
plt.ylabel('Energy + Scaled Wavefunction')
plt.title('Morse Potential: Eigenfunctions and Energy Levels (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Step 6: Select and Normalize the Ground State (n = 0)
# -------------------------
n = 0
psi_ground = eigvecs[:, n]
E_ground = eigvals[n]
psi_ground = psi_ground / np.sqrt(np.sum(np.abs(psi_ground)**2) * dx)

# -------------------------
# Step 7: Animate the Time Evolution of the Ground State
# -------------------------
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, np.real(psi_ground), 'g-', lw=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.5 * np.max(np.abs(psi_ground)), 1.5 * np.max(np.abs(psi_ground)))
ax.set_xlabel('x')
ax.set_ylabel('Re(ψ(x,t))')
ax.set_title(f'Morse Potential Ground State (n=0), E = {E_ground:.2f}')

def update(frame):
    t = frame * 0.05  # Time for the current frame
    psi_t = psi_ground * np.exp(-1j * E_ground * t)
    line.set_ydata(np.real(psi_t))
    return line,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()

