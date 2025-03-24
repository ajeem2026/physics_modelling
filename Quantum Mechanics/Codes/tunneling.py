#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 6: Quantum Tunneling Through a Potential Barrier – Time Evolution Animation with 
Transmission and Reflection Coefficients and Overlay of an Increased Width Potential Barrier

This script simulates quantum tunneling of a Gaussian wave packet through a finite potential barrier.
The barrier is defined as:
    V(x) = V0 for |x| < barrier_width, and V(x) = 0 otherwise,
with the barrier centered at x = 0.
For visualization purposes:
    - V0 is set to 5.0 (making the barrier tall),
    - The barrier width is increased (barrier_width = 5.0, so the barrier extends from –5 to 5),
    - The displayed probability density is scaled by a factor of 20 so that the barrier (the box) appears much taller.
The initial Gaussian wave packet is centered at x0 = -20 with positive momentum so that it moves toward the barrier.
The split-operator method is used for time evolution.
At each time step, the transmission coefficient T (integrated probability for x > barrier_width)
and the reflection coefficient R (integrated probability for x < –barrier_width) are calculated and displayed.
The static potential barrier is overlaid on the probability density plot.

@author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.fft import fft, ifft, fftfreq

# -------------------------
# Step 1: Define the Spatial Grid and Parameters
# -------------------------
N = 1024                      # Number of spatial grid points
x_min, x_max = -50, 50        # Spatial domain from -50 to 50
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]              # Grid spacing

# -------------------------
# Step 2: Define the Potential Barrier
# -------------------------
V0 = 5.0                      # Height of the potential barrier
barrier_width = 5.0           # Increased half-width of the barrier (barrier extends from -5 to 5)
# Define the barrier: V(x) = V0 for |x| < barrier_width, and V(x) = 0 otherwise.
V = np.where(np.abs(x) < barrier_width, V0, 0)

# -------------------------
# Step 3: Define the Initial Gaussian Wave Packet
# -------------------------
x0 = -20.0                    # Initial center (to the left of the barrier)
sigma = 3.0                   # Width of the Gaussian
k0 = 5.0                      # Central momentum (positive, so the wave moves to the right)
psi0 = np.exp(- (x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
# Normalize the wave packet so that ∫|ψ|² dx = 1.
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)

# -------------------------
# Step 4: Define the Momentum Space Grid and Kinetic Operator
# -------------------------
k = fftfreq(N, d=dx) * 2 * np.pi  # Angular frequency for Fourier transform
dt = 0.005                        # Time step for evolution
# Pre-calculate the kinetic evolution operator: T(k) = 0.5 * k² so that operator is exp(-i T dt)
T_op = np.exp(-1j * 0.5 * (k**2) * dt)

# -------------------------
# Step 5: Define the Split-Operator Evolution Function
# -------------------------
def evolve(psi, V, T_op, dt):
    """
    Evolve the wave function psi for one time step dt using the split-operator method.
    """
    psi = np.exp(-1j * V * dt / 2) * psi      # Half-step potential evolution in position space.
    psi_k = fft(psi)                          # Transform to momentum space.
    psi_k = T_op * psi_k                      # Apply kinetic evolution.
    psi = ifft(psi_k)                         # Transform back to position space.
    psi = np.exp(-1j * V * dt / 2) * psi       # Final half-step potential evolution.
    return psi

# -------------------------
# Step 6: Define the Transmission and Reflection Coefficients
# -------------------------
# Define thresholds for transmitted (x > barrier_width) and reflected (x < -barrier_width) regions.
trans_threshold = barrier_width    # x > 5 is considered transmitted.
refl_threshold = -barrier_width      # x < -5 is considered reflected.

def transmission_coefficient(psi, x, threshold):
    """Calculate the transmission coefficient by integrating |ψ|² for x > threshold."""
    return np.sum(np.abs(psi[x > threshold])**2) * dx

def reflection_coefficient(psi, x, threshold):
    """Calculate the reflection coefficient by integrating |ψ|² for x < threshold."""
    return np.sum(np.abs(psi[x < threshold])**2) * dx

# -------------------------
# Step 7: Set Up the Animation
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
# Scaling factor for displaying the wave probability density.
scale_plot = 20  # Lower scaling factor so the wave amplitude remains small.
# Plot the initial (scaled) probability density.
line, = ax.plot(x, scale_plot * np.abs(psi0)**2, 'b-', lw=2, label='Scaled |ψ(x,t)|²')
# Overlay the static potential barrier on the same plot as a dashed black line.
line_barrier, = ax.plot(x, V, 'k--', lw=2, label='Potential Barrier')
ax.set_xlabel('x')
ax.set_ylabel('Scaled Probability Density / Potential')
# Set y-limits such that the potential barrier (V0 = 5.0) appears tall relative to the wave.
ax.set_ylim(0, V0 * 1.5)  # e.g., 0 to 7.5 if V0 = 5.0
ax.set_title('Quantum Tunneling Through a Potential Barrier')
# Create a text object to display time and coefficients.
info_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, color='black')

psi = psi0.copy()  # Initialize the wave packet for simulation.

def update(frame):
    global psi
    # Evolve the wave packet one time step.
    psi = evolve(psi, V, T_op, dt)
    # Compute the actual probability density (without scaling).
    prob_density = np.abs(psi)**2
    # Update the displayed probability density (scaled for visualization).
    line.set_ydata(scale_plot * prob_density)
    # Calculate the transmission coefficient: integrated probability for x > trans_threshold.
    T_coeff = transmission_coefficient(psi, x, trans_threshold)
    # Calculate the reflection coefficient: integrated probability for x < refl_threshold.
    R_coeff = reflection_coefficient(psi, x, refl_threshold)
    current_time = frame * dt
    info_text.set_text(f'Time = {current_time:.3f} s\nTransmission T = {T_coeff:.3f}\nReflection R = {R_coeff:.3f}')
    return line, info_text, line_barrier

# Set the number of time steps (simulate up to t = 1.0 s).
nsteps = int(1.0 / dt)
ani = FuncAnimation(fig, update, frames=nsteps, interval=20, blit=True)

plt.show()
