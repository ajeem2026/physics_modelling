

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:26:00 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to solve the Laplace equation using the Jacobi method
def solve_laplace_jacobi(nx, ny, max_iter=10000, tol=1e-6):
    """
    Solves the 2D Laplace equation using the Jacobi iteration method.
    
    Domain: A unit square [0,1] x [0,1] with the following Dirichlet boundary conditions:
      - Top wall (y = 1) at V = 1.
      - Bottom wall (y = 0) at V = 0.
      - Left (x = 0) and right (x = 1) walls at V = 0.
    
    Parameters:
      nx (int): Number of grid points in the x-direction.
      ny (int): Number of grid points in the y-direction.
      max_iter (int): Maximum number of iterations.
      tol (float): Convergence tolerance based on the maximum change in potential.
    
    Returns:
      phi (ndarray): 2D array of the computed potential.
    """
    # Define grid spacing (assumed uniform)
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    # Initialize the potential array with zeros.
    phi = np.zeros((ny, nx))
    
    # Apply Dirichlet boundary conditions.
    phi[-1, :] = 1.0  # Top wall: V = 1; others remain 0.
    
    # Jacobi iteration loop.
    for iteration in range(max_iter):
        # Copy the current potential to update all interior points simultaneously.
        phi_new = phi.copy()
        
        # Update interior grid points using the average of the four neighboring points.
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                phi_new[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])
        
        # Compute the maximum difference between successive iterations.
        diff = np.max(np.abs(phi_new - phi))
        
        # Update phi for the next iteration.
        phi[:] = phi_new
        
        # Check for convergence.
        if diff < tol:
            print(f'Converged after {iteration+1} iterations')
            break
    else:
        print('Warning: Maximum iterations reached without convergence')
    
    return phi

# Function to compute the analytical solution for the Laplace equation
def analytical_solution(nx, ny, terms=50):
    """
    Computes the analytical solution for Laplace's equation on a unit square with
    the boundary conditions:
      - phi(x,0) = 0, phi(x,1) = 1, phi(0,y) = 0, and phi(1,y) = 0.
    
    The solution is given by the series:
      phi(x,y) = (4/π) * sum_{n odd} [ (1/n) * (sinh(nπy)/sinh(nπ) ) * sin(nπx) ]
    
    Parameters:
      nx (int): Number of grid points in the x-direction.
      ny (int): Number of grid points in the y-direction.
      terms (int): Number of terms to include in the series sum.
    
    Returns:
      phi_analy (ndarray): 2D array of the analytical potential.
    """
    # Create grid coordinates.
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize the analytical solution array.
    phi_analy = np.zeros_like(X)
    
    # Sum over odd n (n=1,3,5,... up to the specified number of terms).
    for n in range(1, terms * 2, 2):
        term = (4/np.pi) * (1/n) * (np.sinh(n * np.pi * Y) / np.sinh(n * np.pi)) * np.sin(n * np.pi * X)
        phi_analy += term
    return phi_analy

# Function to visualize the potential distribution
def plot_potential(phi, title='Potential Distribution'):
    """
    Plots a 2D potential distribution.
    
    Parameters:
      phi (ndarray): 2D array of potential values.
      title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(phi, cmap='inferno', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Potential (V)')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Main part of the code
nx, ny = 50, 50  # Grid size

# Solve the Laplace equation numerically using the Jacobi method.
phi_numerical = solve_laplace_jacobi(nx, ny)

# Compute the analytical solution using the series method.
phi_analytical = analytical_solution(nx, ny, terms=50)

# Compute the absolute error between numerical and analytical solutions.
error = np.abs(phi_numerical - phi_analytical)

# Plot the numerical solution, analytical solution, and the absolute error side-by-side.
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(phi_numerical, cmap='inferno', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Potential (V)')
plt.title('Numerical Solution (Jacobi)')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 3, 2)
plt.imshow(phi_analytical, cmap='inferno', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Potential (V)')
plt.title('Analytical Solution')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 3, 3)
plt.imshow(error, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Absolute Error (V)')
plt.title('Absolute Error')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
