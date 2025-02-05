#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:19:58 2025

@author: mazilui
"""
#This simulates multiple stars interacting gravitationally.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Number of bodies
N = 10  # Number of stars

# Gravitational constant (normalized for simplicity)
G = 1.0

# Initialize positions randomly in 2D space
positions = np.random.uniform(-5, 5, (N, 2))

# Initialize velocities randomly
velocities = np.random.uniform(-1, 1, (N, 2))

# Masses of the bodies (randomized)
masses = np.random.uniform(0.5, 2, N)

# Time step for integration
dt = 0.01
num_steps = 500  # Number of frames in the animation

# Store trajectories
trajectories = np.zeros((num_steps, N, 2))
trajectories[0] = positions

# Function to compute gravitational acceleration
def compute_accelerations(positions, masses):
    accelerations = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = positions[j] - positions[i]
                distance = np.linalg.norm(r_ij) + 1e-5  # Avoid division by zero
                accelerations[i] += G * masses[j] * r_ij / distance**3
    return accelerations

# Velocity Verlet integration
for step in range(1, num_steps):
    accelerations = compute_accelerations(positions, masses)
    positions += velocities * dt + 0.5 * accelerations * dt**2
    new_accelerations = compute_accelerations(positions, masses)
    velocities += 0.5 * (accelerations + new_accelerations) * dt
    trajectories[step] = positions.copy()

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
points, = ax.plot([], [], 'bo')

def update(frame):
    points.set_data(trajectories[frame, :, 0], trajectories[frame, :, 1])
    return points,

ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
plt.show()
