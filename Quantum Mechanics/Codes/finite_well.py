#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3: Finite Potential Well – Energy Levels, Eigenfunctions, and State Animations with Well Shape

This script simulates a finite potential well defined by:
    V(x) = 0 for |x| < a, and V(x) = V0 for |x| >= a.
We perform the following:
  1. Discretize the spatial domain.
  2. Define the finite well potential and its shape.
  3. Construct the Hamiltonian using finite differences.
  4. Solve for the energy eigenvalues and eigenfunctions.
  5. Plot the first four eigenfunctions (each individually scaled and shifted by its energy eigenvalue),
     and overlay the shape of the well (the potential curve).
  6. Animate the time evolution (using the phase factor) of:
      (a) the ground state (n = 0)
      (b) the first excited state (n = 1)

Time evolution for any stationary state is given by:
    ψ(x,t) = ψ(x,0) · exp(-i E t)
which leaves the probability density unchanged, though the real part oscillates.
  
@author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# -------------------------
# PARAMETERS AND DISCRETIZATION
# -------------------------
a = 1.0                    # Half-width of the well (|x| < a defines the well)
V0 = 20.0                  # Barrier height for |x| >= a
N = 500                    # Number of grid points
x_min, x_max = -3, 3       # Spatial domain covering the well and barrier regions
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]           # Grid spacing

# -------------------------
# DEFINE THE FINITE WELL POTENTIAL
# -------------------------
# The potential is defined as:
#    V(x) = 0 for |x| < a (inside the well)
#    V(x) = V0 for |x| >= a (outside the well)
V = np.where(np.abs(x) < a, 0, V0)

# -------------------------
# CONSTRUCT THE HAMILTONIAN USING FINITE DIFFERENCES
# -------------------------
# The kinetic energy operator T is approximated using a central difference:
T = -0.5 * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)) / dx**2
# Total Hamiltonian: H = T + V (with V added as a diagonal matrix)
H = T + np.diag(V)

# -------------------------
# SOLVE THE EIGENVALUE PROBLEM
# -------------------------
eigvals, eigvecs = eigh(H)

# -------------------------
# PLOT THE FIRST FOUR EIGENFUNCTIONS AND THE POTENTIAL SHAPE
# -------------------------
plt.figure(figsize=(10, 6))
for n in range(4):
    # Compute an individual scaling factor so that each eigenfunction's maximum absolute amplitude is 0.5.
    scale = 0.5 / np.max(np.abs(eigvecs[:, n]))
    plt.plot(x, scale * eigvecs[:, n] + eigvals[n], label=f'n = {n}, E = {eigvals[n]:.2f}')
# Overlay the shape of the finite well as a dashed black line.
plt.plot(x, V, 'k--', lw=2, label='Potential Well Shape')
plt.xlabel('x')
plt.ylabel('Energy + Scaled Eigenfunction')
plt.title('Finite Well: Eigenfunctions, Energy Levels, and Well Shape')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# ANIMATION FOR THE GROUND STATE (n = 0)
# =============================================================================
psi_ground = eigvecs[:, 0]
E_ground = eigvals[0]
# Normalize the ground state
psi_ground = psi_ground / np.sqrt(np.sum(np.abs(psi_ground)**2) * dx)

fig1, ax1 = plt.subplots(figsize=(8, 4))
line1, = ax1.plot(x, np.real(psi_ground), 'b-', lw=2)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(-1.5 * np.max(np.abs(psi_ground)), 1.5 * np.max(np.abs(psi_ground)))
ax1.set_xlabel('x')
ax1.set_ylabel('Re(ψ(x,t))')
ax1.set_title(f'Finite Well: Ground State (n=0), E = {E_ground:.2f}')

def update_ground(frame):
    t = frame * 0.05
    psi_t = psi_ground * np.exp(-1j * E_ground * t)
    line1.set_ydata(np.real(psi_t))
    return line1,

ani_ground = FuncAnimation(fig1, update_ground, frames=200, interval=50, blit=True)
plt.show()

# =============================================================================
# ANIMATION FOR THE FIRST EXCITED STATE (n = 1)
# =============================================================================
psi_excited = eigvecs[:, 1]
E_excited = eigvals[1]
# Normalize the first excited state
psi_excited = psi_excited / np.sqrt(np.sum(np.abs(psi_excited)**2) * dx)

fig2, ax2 = plt.subplots(figsize=(8, 4))
line2, = ax2.plot(x, np.real(psi_excited), 'r-', lw=2)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(-1.5 * np.max(np.abs(psi_excited)), 1.5 * np.max(np.abs(psi_excited)))
ax2.set_xlabel('x')
ax2.set_ylabel('Re(ψ(x,t))')
ax2.set_title(f'Finite Well: First Excited State (n=1), E = {E_excited:.2f}')

def update_excited(frame):
    t = frame * 0.05
    psi_t = psi_excited * np.exp(-1j * E_excited * t)
    line2.set_ydata(np.real(psi_t))
    return line2,

ani_excited = FuncAnimation(fig2, update_excited, frames=200, interval=50, blit=True)
plt.show()
