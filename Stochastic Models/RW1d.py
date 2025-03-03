#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:12:03 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
n_steps = 1000            # total number of steps
step_size = 1             # step size (can be negative or positive)
update_interval = 10      # update the animation every 10 steps

# Generate random steps: choose -1 or 1 randomly
steps = np.random.choice([-step_size, step_size], size=n_steps)
# Cumulative sum to generate the random walk
walk = np.cumsum(steps)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, n_steps)
ax.set_ylim(np.min(walk) - 10, np.max(walk) + 10)
ax.set_xlabel('Step')
ax.set_ylabel('Position')
ax.set_title('1D Random Walk')

# Initialize the line object for animation
line, = ax.plot([], [], lw=2, color='blue')

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for the animation
def update(frame):
    xdata = np.arange(frame)
    ydata = walk[:frame]
    line.set_data(xdata, ydata)
    return line,

# Create the animation using FuncAnimation
ani = FuncAnimation(fig, update, frames=range(1, n_steps, update_interval),
                    init_func=init, blit=True, interval=50)

# Display the animation
plt.show()
