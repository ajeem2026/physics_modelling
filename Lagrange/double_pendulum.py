#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:21:53 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravity (m/s^2)
L1, L2 = 1.0, 1.0  # Lengths of pendulums (m)
m1, m2 = 1.0, 1.0  # Masses of pendulums (kg)

# Equations of motion
def equations(t, y):
    theta1, z1, theta2, z2 = y

    delta = theta2 - theta1
    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    denom2 = (L2 / L1) * denom1

    dz1 = (
        m2 * L1 * z1**2 * np.sin(delta) * np.cos(delta) +
        m2 * g * np.sin(theta2) * np.cos(delta) +
        m2 * L2 * z2**2 * np.sin(delta) -
        (m1 + m2) * g * np.sin(theta1)
    ) / denom1

    dz2 = (
        -L2 / L1 * z2**2 * np.sin(delta) * np.cos(delta) +
        (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
        (m1 + m2) * L1 * z1**2 * np.sin(delta) -
        (m1 + m2) * g * np.sin(theta2)
    ) / denom2

    return [z1, dz1, z2, dz2]

# Initial conditions: [theta1, omega1, theta2, omega2]
theta1_0, theta2_0 = np.radians(120), np.radians(-10)  # Initial angles
omega1_0, omega2_0 = 0.0, 0.0  # Initial angular velocities
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

# Time span
t_span = (0, 10)  # Simulate for 10 seconds
t_eval = np.linspace(*t_span, 1000)  # 1000 time steps

# Solve the equations
solution = solve_ivp(equations, t_span, y0, t_eval=t_eval, method="RK45")

# Extract results
theta1, theta2 = solution.y[0], solution.y[2]

# Convert to Cartesian coordinates
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Animation setup
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

line, = ax.plot([], [], 'o-', lw=2)
trace_x, trace_y = [], []

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    trace_x.append(x2[i])
    trace_y.append(y2[i])
    ax.plot(trace_x, trace_y, 'r', alpha=0.5)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=10, blit=True)
plt.show()
