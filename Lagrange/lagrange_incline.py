"""
Block on a Frictionless Incline Using Lagrange Equations

This script derives and solves the equation of motion for a block sliding down a 
frictionless inclined plane using Lagrangian mechanics. The solution is computed 
both analytically and symbolically using SymPy, and the results are visualized 
using Matplotlib.

Author: Abid Jeem

Functions:
    - lagrangian_equation(): Derives the equation of motion using Lagrange’s equation.
    - solve_equation(): Solves the second-order differential equation symbolically.
    - analytical_solution(): Computes the analytical solution via direct integration.
    - compare_solutions(): Verifies that the symbolic and analytical solutions match.
    - simulate_motion(): Performs a numerical simulation of the block’s motion.
    - plot_results(): Plots and compares the analytical and symbolic solutions.

Returns:
    - Equation of motion derived using the Euler-Lagrange formalism.
    - Symbolic and analytical solutions for the position x(t).
    - A numerical simulation showing the block’s trajectory.
    - A plot comparing the two solutions.

Usage:
    Run this script to generate symbolic and numerical solutions for the problem 
    and visualize the results.

"""
import sympy as sp  # Import SymPy for symbolic calculations
import numpy as np  # Import NumPy for numerical evaluations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Define time variable
t = sp.Symbol('t', real=True)

# Define physical constants: mass (m), gravity (g), and incline angle (alpha)
m, g, alpha = sp.symbols('m g alpha', real=True, positive=True)

# Define position function x(t) symbolically
x = sp.Function('x')(t)

# Define kinetic energy (T) of the block
T = (1/2) * m * sp.diff(x, t)**2  # (1/2) m v^2

# Define potential energy (U) of the block (taking zero potential at the top)
U = -m * g * x * sp.sin(alpha)  # U = -mgx sin(alpha)

# Define the Lagrangian as L = T - U
L = T - U

# Compute Euler-Lagrange equation
dL_dx = sp.diff(L, x)  # Partial derivative of L with respect to x
dL_dxdot = sp.diff(L, sp.diff(x, t))  # Partial derivative of L with respect to x-dot
d_dt = sp.diff(dL_dxdot, t)  # Time derivative of the above term

# Equation of motion (EOM) derived from Euler-Lagrange equation
eom = sp.Eq(d_dt - dL_dx, 0)

# Solve the differential equation symbolically
sol = sp.dsolve(eom, x)

# Print the equation of motion
print("Equation of motion:")
sp.pprint(eom)

# Print the symbolic solution
print("\nSymbolic Solution:")
sp.pprint(sol)

# Define the analytical solution manually
C1, C2 = sp.symbols('C1 C2')  # Integration constants
analytical_sol = C1 + C2 * t + (1/2) * g * sp.sin(alpha) * t**2  # Expected solution

# Print the analytical solution
print("\nAnalytical Solution:")
sp.pprint(analytical_sol)

# Compare the symbolic and analytical solutions
comparison = sp.simplify(sol.rhs - analytical_sol)
print("\nDifference between symbolic and analytical solutions:")
sp.pprint(comparison)  # Should be zero, confirming correctness

# Numerical Simulation Parameters
g_val = 9.81  # Acceleration due to gravity in m/s^2
alpha_val = np.radians(30)  # Convert 30 degrees to radians

# Define the numerical function for x(t) based on the analytical solution
def x_func(t):
    return 0.5 * g_val * np.sin(alpha_val) * t**2  # x(t) = (1/2) g sin(alpha) * t^2

# Generate time values for plotting
time_vals = np.linspace(0, 5, 100)  # Time from 0 to 5 seconds, 100 points
x_vals = x_func(time_vals)  # Compute corresponding x values

# Convert symbolic solution to numerical function using lambdify
x_symbolic = sp.lambdify(t, sol.rhs.subs({g: g_val, alpha: alpha_val, C1: 0, C2: 0}), "numpy")
x_symbolic_vals = x_symbolic(time_vals)  # Compute symbolic solution numerically

# Plot the results
plt.figure(figsize=(8, 5))  # Set figure size
plt.plot(time_vals, x_vals, label=r'Analytical $x(t) = \frac{1}{2} g \sin \alpha \cdot t^2$', color='b', linestyle='dashed')
plt.plot(time_vals, x_symbolic_vals, label='Symbolic Solution from Lagrangian', color='r')
plt.xlabel("Time (s)")  # Label x-axis
plt.ylabel("Position x (m)")  # Label y-axis
plt.title("Block Sliding Down a Frictionless Incline: Analytical vs Symbolic Solution")  # Set title
plt.legend()  # Display legend
plt.grid()  # Show grid
plt.show()  # Display the plot
