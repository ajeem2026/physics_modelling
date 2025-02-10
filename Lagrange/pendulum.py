#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:06:17 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# ------------------------------------------------------------
# 1) SYMBOLIC DERIVATION OF PENDULUM ODE USING THE LAGRANGIAN
# ------------------------------------------------------------

# Define symbolic variables
t = sp.Symbol('t', real=True)         # time
theta = sp.Function('theta')(t)       # angular displacement, theta(t)
L, m, g = sp.symbols('L m g', real=True, positive=True)

# Kinetic energy (T): mass m, length L => velocity = L * dtheta/dt
dtheta = theta.diff(t)
T = 0.5 * m * (L * dtheta)**2

# Potential energy (V): choose V=0 at pivot => best to measure relative to lowest point
# A common choice: V = m g L (1 - cos(theta))
V = m * g * L * (1 - sp.cos(theta))

# Lagrangian Lagr = T - V
Lagr = T - V

# Euler-Lagrange:
# d/dt(∂L/∂dot(theta)) - ∂L/∂theta = 0
dL_dtheta = Lagr.diff(theta)
dL_ddtheta = Lagr.diff(dtheta)
eq_expr = (dL_ddtheta.diff(t) - dL_dtheta).simplify()

# eq_expr should give us something ~> m L^2 d2theta/dt^2 + m g L sin(theta) = 0
# We'll form the equation eq_expr = 0 for dsolve
ode = sp.Eq(eq_expr, 0)

# Solve symbolically for theta(t). Usually yields a complicated expression,
# but let's just confirm the ODE in standard form:
print("\nSymbolic ODE from Lagrangian:")
sp.pprint(eq_expr)

# ------------------------------------------------------------
# 2) NUMERICAL SOLUTION OF THE PENDULUM EQUATION
# ------------------------------------------------------------

# The equation of motion from the Lagrangian:
# m L^2 d2theta/dt^2 + m g L sin(theta) = 0
# => d2theta/dt^2 + (g/L) sin(theta) = 0
# Let's define an ODE system in first-order form:
# y1 = theta, y2 = dtheta/dt
# => y1' = y2
# => y2' = - (g/L) sin(y1)

def pendulum_equations(t, y, length, gravity):
    # y[0] = theta, y[1] = dtheta/dt
    theta_val, dtheta_val = y
    d2theta_val = - (gravity / length) * np.sin(theta_val)
    return [dtheta_val, d2theta_val]

# Set physical constants
length = 1.0  # meters
mass = 1.0    # kg (not needed in the final ODE, but we keep it for clarity)
gravity = 9.81

# Initial conditions: [theta(0), dtheta/dt(0)]
# Example: 45 degrees offset, zero initial angular velocity
theta0 = np.radians(45.0)
omega0 = 0.0
y0 = [theta0, omega0]

# Time span for the simulation
t_span = (0, 10)             # simulate 10 seconds
t_eval = np.linspace(0, 10, 300)  # time points to store solution

# Solve with solve_ivp
solution = solve_ivp(
    pendulum_equations,
    t_span,
    y0,
    t_eval=t_eval,
    args=(length, gravity),
    method='RK45'
)

theta_sol = solution.y[0,:]
time_sol = solution.t

# ------------------------------------------------------------
# 3) ANIMATION OF THE PENDULUM
# ------------------------------------------------------------

# Convert polar (theta) to Cartesian for visualization
# Pivot at origin (0,0), mass at x=L sin(theta), y=-L cos(theta)
x_sol = length * np.sin(theta_sol)
y_sol = -length * np.cos(theta_sol)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.2*length, 1.2*length)
ax.set_ylim(-1.2*length, 0.2*length)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Pendulum Animation (Lagrangian Approach)")
ax.grid(True)

# Pendulum rod + bob
line, = ax.plot([], [], 'o-', lw=2, color='darkblue')
time_template = ax.text(
    0.05, 0.90, '', transform=ax.transAxes, fontsize=12, color='red'
)

def init():
    line.set_data([], [])
    time_template.set_text('')
    return line, time_template

def animate(i):
    # i is the frame index
    x = [0, x_sol[i]]  # pivot to mass
    y = [0, y_sol[i]]
    line.set_data(x, y)
    time_template.set_text(f"t = {time_sol[i]:.2f} s")
    return line, time_template

ani = animation.FuncAnimation(fig, animate, frames=len(time_sol),
                              interval=30, init_func=init, blit=True)

plt.show()
