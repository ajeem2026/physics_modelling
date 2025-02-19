#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:28:09 2025

@author: mazilui
"""

#!/usr/bin/env python3
"""
Animated FTCS Method for Solving the 1D Heat Equation

This script uses the Forward Time Centered Space (FTCS) method to solve
the one-dimensional heat equation:
    ∂u/∂t = α ∂²u/∂x²
on the domain x ∈ [0, 1] with Dirichlet boundary conditions:
    u(0,t) = 0 and u(1,t) = 0,
and initial condition:
    u(x,0) = sin(πx)

The animation shows how the solution evolves over time.
Author: Your Name
Date: Today's Date
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
alpha = 1.0       # thermal diffusivity
L = 1.0           # length of the rod
T = 0.5           # total simulation time
nx = 50           # number of spatial grid points
dx = L / (nx - 1) # spatial step size
dt = 0.0001       # time step size (choose dt such that alpha*dt/dx^2 <= 0.5)
nt = int(T / dt)  # total number of time steps

# Stability parameter lambda
lam = alpha * dt / dx**2
if lam > 0.5:
    print("Warning: Stability condition violated, lam =", lam)

# Create spatial grid and initial condition: u(x,0) = sin(πx)
x = np.linspace(0, L, nx)
u = np.sin(np.pi * x)
u_new = np.zeros(nx)

# Set up the animation: choose the number of frames and steps per frame
frames = 200                   # total number of frames in the animation
steps_per_frame = nt // frames # simulation steps per animation frame

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x, u, label="Numerical Solution")
ax.set_xlim(0, L)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Heat Equation Animation (FTCS)")
ax.legend()

def update(frame):
    """
    Update function for the animation.
    Advances the simulation by a fixed number of steps and updates the plot.
    """
    global u, u_new
    # Perform several time steps per frame for efficiency
    for _ in range(steps_per_frame):
        # Update interior points using FTCS scheme
        u_new[1:-1] = u[1:-1] + lam * (u[2:] - 2*u[1:-1] + u[:-2])
        # Enforce Dirichlet boundary conditions: u(0)=u(L)=0
        u_new[0] = 0
        u_new[-1] = 0
        # Update solution for the next time step
        u = u_new.copy()
    # Update the line data for the plot
    line.set_ydata(u)
    return line,

# Create the animation
anim = FuncAnimation(fig, update, frames=frames, blit=True)

# Display the animation window
plt.show()
