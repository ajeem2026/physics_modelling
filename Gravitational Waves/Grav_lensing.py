#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:21:28 2025

@author: mazilui+ChatGPT
"""
#This simulates how a black hole bends light.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grid size
grid_size = 100
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)

# Schwarzschild radius (for visualization)
r_s = 1.0

# Compute radial distance from black hole
R = np.sqrt(X**2 + Y**2)

# Gravitational distortion effect
def lensing_effect(R):
    return np.exp(-R/r_s)

# Animation function
fig, ax = plt.subplots()
im = ax.imshow(lensing_effect(R), cmap='inferno', extent=(-5,5,-5,5))

def update(frame):
    new_R = R - 0.01 * frame  # Simulating gravitational wave motion
    im.set_array(lensing_effect(new_R))
    return im,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()
