#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:25:12 2025

@author: mazilui+ChatGPT
"""
#This shows the gravitational wave pattern as two black holes merge.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grid definition
grid_size = 100
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)

# Generate wave pattern
def wave_pattern(frame):
    R = np.sqrt(X**2 + Y**2)
    return np.sin(2 * np.pi * (R - frame * 0.1)) * np.exp(-0.2 * R)

# Animation
fig, ax = plt.subplots()
im = ax.imshow(wave_pattern(0), cmap='inferno', extent=(-5,5,-5,5))

def update(frame):
    im.set_array(wave_pattern(frame))
    return im,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()
