#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:23:41 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 1.0
m1, m2 = 1.0, 2.0  # Masses
r = 2.0  # Separation
dt = 0.05
num_steps = 300

# Initial positions (opposite sides of orbit)
positions = np.array([[r * m2 / (m1 + m2), 0], [-r * m1 / (m1 + m2), 0]])

# Initial velocities (perpendicular to position vectors)
v1 = np.sqrt(G * m2 / r)
v2 = np.sqrt(G * m1 / r)
velocities = np.array([[0, v1], [0, -v2]])

# Store trajectories
trajectories = np.zeros((num_steps, 2, 2))

# Velocity Verlet Integration
for step in range(num_steps):
    trajectories[step] = positions
    r_vec = positions[1] - positions[0]
    r_mag = np.linalg.norm(r_vec)
    acc1 = G * m2 * r_vec / r_mag**3
    acc2 = -G * m1 * r_vec / r_mag**3
    positions += velocities * dt + 0.5 * np.array([acc1, acc2]) * dt**2
    velocities += np.array([acc1, acc2]) * dt

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
points, = ax.plot([], [], 'bo')

def update(frame):
    points.set_data(trajectories[frame, :, 0], trajectories[frame, :, 1])
    return points,

ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=20, blit=True)
plt.show()
