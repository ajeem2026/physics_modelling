#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:16:31 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
n_steps = 1000            # Total number of steps
update_interval = 10      # Update the animation every 10 steps
bias_probability = [0.4, 0.1, 0.1, 0.4]  # Biased probabilities for [right, left, up, down]

# Define possible moves: up, down, left, right
moves = np.array([
    [1, 0],   # move right
    [-1, 0],  # move left
    [0, 1],   # move up
    [0, -1]   # move down
])

# Choose a random move for each step with biased probabilities
random_indices = np.random.choice(len(moves), size=n_steps, p=bias_probability)
steps = moves[random_indices]

# Compute the cumulative sum to get the (x, y) coordinates of the random walk
walk = np.cumsum(steps, axis=0)
x = walk[:, 0]
y = walk[:, 1]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(np.min(x) - 5, np.max(x) + 5)
ax.set_ylim(np.min(y) - 5, np.max(y) + 5)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('2D Random Walk with Bias (Higher Probability for Right and Left)')

# Initialize the line and point objects for animation
line, = ax.plot([], [], lw=2, color='blue')
point, = ax.plot([], [], 'ro', markersize=6)

# Initialization function for the animation
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# Update function for the animation
def update(frame):
    # Get the coordinates up to the current frame
    current_x = x[:frame]
    current_y = y[:frame]
    line.set_data(current_x, current_y)
    # Wrap the current point in lists to satisfy set_data requirements
    point.set_data([current_x[-1]], [current_y[-1]])
    return line, point

# Create the animation using FuncAnimation (with blit=False for compatibility)
ani = FuncAnimation(
    fig, update, frames=np.arange(1, n_steps, update_interval),
    init_func=init, blit=False, interval=50
)

plt.show()