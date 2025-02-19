

#!/usr/bin/env python3
"""
1D Wave Equation Solver with Animation

This script numerically solves the one-dimensional wave equation:
    ∂²u/∂t² = c² ∂²u/∂x²
using a finite difference scheme and animates the solution over time.

Discretization:
--------------
We discretize the spatial domain [0, L] into nx points and use a time step dt.
The finite difference update formula for the wave equation is given by:
    u_i^(n+1) = 2u_i^n - u_i^(n-1) + (c*dt/dx)² (u_(i+1)^n - 2u_i^n + u_(i-1)^n)
where:
    - u_i^n approximates u(x_i, t^n),
    - λ = c dt/dx is the CFL number, which should satisfy λ ≤ 1 for stability.

Initial and Boundary Conditions:
---------------------------------
- Initial displacement: a Gaussian pulse centered in the domain.
- Initial velocity: zero.
- Dirichlet boundary conditions: u(0,t) = u(L,t) = 0.

Animation:
----------
Matplotlib’s FuncAnimation is used to animate the solution, updating the plot
after a fixed number of time steps per frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the wave equation
c = 1.0            # Wave speed
L = 1.0            # Length of the spatial domain [0, L]
nx = 200           # Number of spatial grid points
dx = L / (nx - 1)  # Spatial step size

T = 2.0            # Total simulation time
dt = 0.0005        # Time step size (choose dt such that λ = c*dt/dx ≤ 1 for stability)
nt = int(T / dt)   # Total number of time steps

# Compute the CFL number (λ = c*dt/dx)
lambda_val = c * dt / dx
print("CFL number (lambda) =", lambda_val)
if lambda_val > 1.0:
    print("Warning: CFL condition violated; the simulation may be unstable!")

# Define the spatial grid
x = np.linspace(0, L, nx)

# Define initial conditions:
def initial_displacement(x):
    """Initial displacement: Gaussian pulse centered at x = 0.5."""
    return np.exp(-100 * (x - 0.5)**2)

def initial_velocity(x):
    """Initial velocity: zero everywhere."""
    return 0.0

# Set initial displacement u(x,0)
u0 = initial_displacement(x)

# Initialize u at time step 1 (using a Taylor expansion with zero initial velocity)
u1 = np.zeros(nx)
# Use finite difference to compute u1 at interior points
u1[1:-1] = (u0[1:-1] +
            dt * initial_velocity(x[1:-1]) +
            0.5 * (lambda_val**2) * (u0[2:] - 2*u0[1:-1] + u0[:-2]))
# Enforce boundary conditions at t = 0 and t = dt
u0[0] = u0[-1] = 0
u1[0] = u1[-1] = 0

# Prepare arrays for the time-stepping scheme:
u_prev = u0.copy()  # u at time step n-1 (initial time)
u_curr = u1.copy()  # u at time step n (first time step)

# Set up the plot for animation
fig, ax = plt.subplots()
line, = ax.plot(x, u0, lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("1D Wave Equation Animation")

# Determine animation parameters:
frames = 400                    # Total number of frames in the animation
steps_per_frame = nt // frames  # Number of time steps per animation frame

def update(frame):
    """
    Update function for the animation.
    Advances the simulation by 'steps_per_frame' time steps and updates the plot.
    """
    global u_prev, u_curr
    for _ in range(steps_per_frame):
        # Compute next time step using the finite difference scheme
        u_next = np.zeros(nx)
        # Update interior points using the wave equation discretization
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] +
                        (lambda_val**2) * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))
        # Enforce Dirichlet boundary conditions: u(0) = u(L) = 0
        u_next[0] = 0
        u_next[-1] = 0
        # Shift the time levels: u_prev <- u_curr, u_curr <- u_next
        u_prev, u_curr = u_curr, u_next
    # Update the line object for the animation
    line.set_ydata(u_curr)
    return line,

# Create the animation using FuncAnimation
anim = FuncAnimation(fig, update, frames=frames, blit=True)

# Display the animation
plt.show()
