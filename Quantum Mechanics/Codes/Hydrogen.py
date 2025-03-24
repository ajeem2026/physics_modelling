#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5: Hydrogen Atom Orbitals – Combined Visualization of 1s and 2p₋z Orbitals

This script visualizes two hydrogen atom orbitals:
  (a) The 1s orbital, defined by:
      ψ₁s(r) = 1/√π · exp(−r),
      with probability density |ψ₁s(r)|² = 1/π · exp(−2r),
      plotted on a 2D slice in the x–y plane (z = 0).
  (b) The 2p₋z orbital, defined by:
      ψ₂p_z(x,z) = 1/(4√(2π)) · z · exp(−r/2), where r = √(x² + z²),
      with its probability density |ψ₂p_z(x,z)|²,
      plotted on a 2D slice in the x–z plane (y = 0).

Both orbitals are displayed in a single figure using subplots.
A simple animation is set up that updates the titles to simulate dynamism.

@author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# Hydrogen 1s Orbital (x-y plane, z = 0)
# -------------------------
N1 = 200                     # Number of grid points along each axis for 1s orbital
xy_min, xy_max = -10, 10     # Spatial domain limits for x and y
x = np.linspace(xy_min, xy_max, N1)
y = np.linspace(xy_min, xy_max, N1)
X, Y = np.meshgrid(x, y)     # Create a meshgrid for the x-y plane
R = np.sqrt(X**2 + Y**2)       # Radial distance from the origin

# 1s orbital wave function and probability density
psi_1s = (1/np.sqrt(np.pi)) * np.exp(-R)
density_1s = np.abs(psi_1s)**2  # |ψ₁s|² = (1/π) exp(-2r)

# -------------------------
# Hydrogen 2p₋z Orbital (x-z plane, y = 0)
# -------------------------
N2 = 200                     # Number of grid points along each axis for 2p₋z orbital
xz_min, xz_max = -10, 10     # Spatial domain limits for x and z
x2 = np.linspace(xz_min, xz_max, N2)
z2 = np.linspace(xz_min, xz_max, N2)
X2, Z2 = np.meshgrid(x2, z2)  # Create a meshgrid for the x-z plane
R2 = np.sqrt(X2**2 + Z2**2)    # Radial distance in the x-z plane

# 2p₋z orbital wave function and probability density
prefactor = 1/(4 * np.sqrt(2 * np.pi))
psi_2pz = prefactor * Z2 * np.exp(-R2/2)
density_2pz = np.abs(psi_2pz)**2

# -------------------------
# Create Figure with Two Subplots
# -------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot the Hydrogen 1s Orbital on the left subplot (x-y plane)
cont1 = ax1.contourf(X, Y, density_1s, 100, cmap='inferno')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Hydrogen 1s Orbital Probability Density (z=0)')
cbar1 = plt.colorbar(cont1, ax=ax1)
cbar1.set_label('Probability Density')

# Plot the Hydrogen 2p₋z Orbital on the right subplot (x-z plane)
cont2 = ax2.contourf(X2, Z2, density_2pz, 100, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_title('Hydrogen 2p$_z$ Orbital Probability Density (y=0)')
cbar2 = plt.colorbar(cont2, ax=ax2)
cbar2.set_label('Probability Density')

# -------------------------
# Animation: Update Titles to Simulate Dynamism
# -------------------------
def update(frame):
    # Update titles with the current frame number
    ax1.set_title(f'Hydrogen 1s Orbital (Static) - Frame {frame}')
    ax2.set_title(f'Hydrogen 2p$_z$ Orbital (Static) - Frame {frame}')
    return ax1, ax2

ani = FuncAnimation(fig, update, frames=100, interval=100, blit=False)

plt.tight_layout()
plt.show()
