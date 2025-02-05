#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:49:52 2025

@author: mazilui+ChatGPT
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define gravitational constant (normalized for simplicity)
G = 1.0

# Define masses of the three bodies
m1, m2, m3 = 1.0, 1.0, 1.0  # Assume equal masses for simplicity

# Define initial positions and velocities for the figure-eight solution
positions = np.array([
    [0.97000436, -0.24308753, 0.0],   # Body 1
    [-0.97000436, 0.24308753, 0.0],  # Body 2
    [0.0, 0.0, 0.0]    # Body 3
], dtype=np.float64)

velocities = np.array([
    [0.4662036850, 0.4323657300, 0.0],   # Body 1
    [0.4662036850, 0.4323657300, 0.0],  # Body 2
    [-0.93240737, -0.86473146, 0.0]   # Body 3
], dtype=np.float64)

masses = np.array([m1, m2, m3])

# Function to compute accelerations due to gravity
def compute_accelerations(positions, masses):
    n = len(masses)
    accelerations = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_ij = positions[j] - positions[i]  # Distance vector
                distance = np.linalg.norm(r_ij) + 1e-5  # Avoid division by zero
                accelerations[i] += G * masses[j] * r_ij / distance**3
    return accelerations

# Time parameters
dt = 0.01  # Time step
num_steps = 1000  # Increased number of steps for smoother animation

# Store trajectories
trajectories = np.zeros((num_steps, 3, 3))  # (time, body, dimension)
trajectories[0] = positions

# Velocity Verlet integration loop
for step in range(1, num_steps):
    accelerations = compute_accelerations(positions, masses)
    positions += velocities * dt + 0.5 * accelerations * dt**2
    new_accelerations = compute_accelerations(positions, masses)
    velocities += 0.5 * (accelerations + new_accelerations) * dt
    trajectories[step] = positions.copy()

# Visualization with animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']
labels = ['Body 1', 'Body 2', 'Body 3']
points = [ax.plot([], [], [], 'o', color=colors[i])[0] for i in range(3)]
trails = [ax.plot([], [], [], '-', color=colors[i])[0] for i in range(3)]

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Figure-Eight Three-Body Problem Animation')

# Animation function
def update(frame):
    for i in range(3):
        points[i].set_data([trajectories[frame, i, 0]], [trajectories[frame, i, 1]])
        points[i].set_3d_properties([trajectories[frame, i, 2]])
        trails[i].set_data(trajectories[:frame, i, 0], trajectories[:frame, i, 1])
        trails[i].set_3d_properties(trajectories[:frame, i, 2])
    return points + trails

ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
plt.show()
