#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:27:32 2025

@author: mazilui+ChatGPT
"""
#This simulation shows how light from stars shifts in wavelength as an observer moves at relativistic speeds.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
c = 1.0  # Speed of light (normalized)
lambda_rest = np.linspace(400, 700, 100)  # Visible light wavelengths in nm

# Function to compute relativistic Doppler shift
def doppler_shift(velocity):
    return lambda_rest * np.sqrt((1 - velocity/c) / (1 + velocity/c))

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(400, 700)
ax.set_ylim(0, 1)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity")
ax.set_title("Relativistic Doppler Effect")

# Initial spectrum (normalized intensity)
spectrum, = ax.plot(lambda_rest, np.exp(-0.002 * (lambda_rest - 550)**2), color='b')

# Update function for animation
def update(frame):
    velocity =0.2 * (frame / 100)  # Increasing velocity up to 90% speed of light
    lambda_shifted = doppler_shift(velocity)
    spectrum.set_xdata(lambda_shifted)  # Shift spectrum
    return spectrum,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.show()

