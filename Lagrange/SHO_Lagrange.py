#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:10:19 2025

@author: mazilui+ChatGPT
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1) Define symbols and the unknown function q1(t)
t = sp.Symbol('t', real=True)
m, k = sp.symbols('m k', positive=True)  # Assume m>0, k>0 so sqrt(k/m) is well-defined
q1 = sp.Function('q1')(t)

# 2) Define the Lagrangian: L = T - V
dq1 = q1.diff(t)
T = 0.5 * m * dq1**2                # Kinetic energy
V = 0.5 * k * q1**2                 # Potential energy
L = T - V

# 3) Euler-Lagrange equation: d/dt(∂L/∂(dq1)) - ∂L/∂q1 = 0
dL_dq = sp.diff(L, q1)              # ∂L/∂q1
dL_ddq = sp.diff(L, dq1)            # ∂L/∂(dq1)
eq_expr = sp.diff(dL_ddq, t) - dL_dq  # d/dt(∂L/∂(dq1)) - ∂L/∂q1

# 4) Simplify and solve the ODE: m*q1'' + k*q1 = 0
eq_simpl = sp.simplify(eq_expr)     # Usually yields m*q1'' + k*q1
ode = sp.Eq(eq_simpl, 0)            # Make it eq(...) = 0 form
solution = sp.dsolve(ode, q1)

print("ODE from Lagrangian:")
sp.pprint(eq_simpl)
print("\nGeneral Solution:")
sp.pprint(solution)

# 5) Visualize a numeric example (m=1, k=1) for q1(t).
#    We'll manually pick integration constants C1=1, C2=0 for demonstration.

# Convert solution to a usable python function
# The general solution is q1(t) = C1*cos( sqrt(k/m)*t ) + C2*sin( sqrt(k/m)*t )
# We'll pick C1=1, C2=0 for plotting:
omega = np.sqrt(1.0/1.0)  # sqrt(k/m) with k=1, m=1
t_vals = np.linspace(0, 10, 300)
q_vals = np.cos(omega * t_vals)  # C1=1, C2=0

# Plot
plt.figure(figsize=(7, 4))
plt.plot(t_vals, q_vals, label="q1(t) = cos(ωt)")
plt.title("Simple Harmonic Oscillator (m=1, k=1)")
plt.xlabel("Time (s)")
plt.ylabel("Position q1(t)")
plt.grid(True)
plt.legend()
plt.show()
