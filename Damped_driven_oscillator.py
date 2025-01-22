#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:18:52 2025

@author: mazilui+ChatGPT
"""

# Python Code: Oscillations Analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Simple Harmonic Oscillator Function
def simple_harmonic_oscillator(y, t, m, k):
    """
    Defines the system of ODEs for a simple harmonic oscillator.
    """
    x, v = y
    dydt = [v, -k / m * x]
    return dydt

# Damped Harmonic Oscillator Function
def damped_oscillator(y, t, m, k, b):
    """
    Defines the system of ODEs for a damped harmonic oscillator.
    """
    x, v = y
    dydt = [v, -b / m * v - k / m * x]
    return dydt

# Driven Damped Harmonic Oscillator Function
def driven_damped_oscillator(y, t, m, k, b, F, omega):
    """
    Defines the system of ODEs for a driven damped harmonic oscillator.
    """
    x, v = y
    dydt = [v, -b / m * v - k / m * x + F / m * np.cos(omega * t)]
    return dydt

# Parameters
m = 1.0  # Mass (kg)
k = 10.0  # Spring constant (N/m)
x0 = 1.0  # Initial position (m)
v0 = 0.0  # Initial velocity (m/s)
t_end = 20.0  # Simulation time (s)
dt = 0.01  # Time step (s)
t = np.arange(0, t_end, dt)

# Case 1: No Damping, No External Force
y0 = [x0, v0]
solution = odeint(simple_harmonic_oscillator, y0, t, args=(m, k))
x = solution[:, 0]
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='No Damping, No External Force')
plt.title('No Damping, No External Force')
plt.xlabel('Time (s)')
plt.ylabel('Position x(t)')
plt.legend()
plt.grid()
plt.show()

# Case 2: Damped Oscillator
b_values = [1.0, 2.0, 10.0]  # Underdamped, Critically damped, Overdamped
labels = ['Underdamped', 'Critically Damped', 'Overdamped']
plt.figure(figsize=(10, 6))
for b, label in zip(b_values, labels):
    solution = odeint(damped_oscillator, y0, t, args=(m, k, b))
    x = solution[:, 0]
    plt.plot(t, x, label=f'{label}: b={b}')
plt.title('Damped Oscillator: Underdamped, Critically Damped, Overdamped')
plt.xlabel('Time (s)')
plt.ylabel('Position x(t)')
plt.legend()
plt.grid()
plt.show()

# Case 3: Driven Damped Oscillator
b = 1.0  # Damping coefficient (kg/s)
F = 5.0  # Driving force amplitude (N)
omega = 1.5  # Driving angular frequency (rad/s) (chosen to show transient behavior)
solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
x = solution[:, 0]
plt.figure(figsize=(10, 6))
plt.plot(t, x, label=f'Driven Damped: F={F}, omega={omega}')
plt.title('Driven Damped Oscillator with Transient Behavior')
plt.xlabel('Time (s)')
plt.ylabel('Position x(t)')
plt.legend()
plt.grid()
plt.show()
