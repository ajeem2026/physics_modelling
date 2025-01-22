#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:29:38 2025

@author: mazilui+CHATGPT
"""
# This one is without a driven force 
# Python Code: Simple Harmonic Oscillator using ODEINT
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def simple_harmonic_oscillator(y, t, m, k):
    """
    Defines the system of ODEs for a simple harmonic oscillator.
    Args:
        y: Array of [x, v], where x is position and v is velocity.
        t: Time (s)
        m: Mass (kg)
        k: Spring constant (N/m)
    Returns:
        dydt: Derivatives [dx/dt, dv/dt].
    """
    # Unpack the position (x) and velocity (v) from the input array
    x, v = y
    # Define the derivatives dx/dt = v and dv/dt = -k/m * x
    dydt = [v, -k / m * x]
    return dydt

# Parameters for the simple harmonic oscillator
m = 1.0  # Mass (kg)
k = 10.0  # Spring constant (N/m)
x0 = 1.0  # Initial position (m)
v0 = 0.0  # Initial velocity (m/s)
t_end = 10.0  # Total simulation time (s)
dt = 0.01  # Time step (s)

# Create a time array from 0 to t_end with increments of dt
t = np.arange(0, t_end, dt)

# Initial conditions: position x0 and velocity v0
y0 = [x0, v0]

# Solve the ODE using odeint
# The solution y contains position and velocity at each time step
y = odeint(simple_harmonic_oscillator, y0, t, args=(m, k))
# Extract position (x) and velocity (v) from the solution
x, v = y[:, 0], y[:, 1]

# Plot the results for position and velocity
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Position x(t)')
plt.plot(t, v, label='Velocity v(t)', linestyle='--')
plt.title('Simple Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# Python Code: Damped Oscillator using ODEINT
def damped_oscillator(y, t, m, k, b):
    """
    Defines the system of ODEs for a damped harmonic oscillator.
    Args:
        y: Array of [x, v], where x is position and v is velocity.
        t: Time (s)
        m: Mass (kg)
        k: Spring constant (N/m)
        b: Damping coefficient (kg/s)
    Returns:
        dydt: Derivatives [dx/dt, dv/dt].
    """
    # Unpack the position (x) and velocity (v) from the input array
    x, v = y
    # Define the derivatives dx/dt = v and dv/dt = -b/m * v - k/m * x
    dydt = [v, -b / m * v - k / m * x]
    return dydt

# Parameters for the damped oscillator
b_values = [0.5, 2.0, 10.0]  # Damping coefficients: Underdamped, Critically damped, Overdamped
labels = ['Underdamped', 'Critically Damped', 'Overdamped']

# Initialize a plot for all damping cases
plt.figure(figsize=(10, 6))
for b, label in zip(b_values, labels):
    # Initial conditions: position x0 and velocity v0
    y0 = [x0, v0]

    # Solve the ODE using odeint for each damping case
    y = odeint(damped_oscillator, y0, t, args=(m, k, b))
    # Extract position (x) from the solution
    x = y[:, 0]

    # Plot the position for the current damping case
    plt.plot(t, x, label=f'{label}: b={b}')

# Finalize the plot
plt.title('Damped Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Position x(t)')
plt.legend()
plt.grid()
plt.show()
