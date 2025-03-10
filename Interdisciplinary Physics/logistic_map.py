#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:10:13 2025

@author: mazilui+ChatGPT
"""
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x0, n_iter):
    """
    Compute iterations of the logistic map:
      x_{n+1} = r * x_n * (1 - x_n)
    
    Parameters:
        r (float): Control parameter (growth rate)
        x0 (float): Initial condition (0 <= x0 <= 1)
        n_iter (int): Number of iterations
        
    Returns:
        np.ndarray: Array of x values from 0 to n_iter-1
    """
    x = np.empty(n_iter)
    x[0] = x0
    for i in range(1, n_iter):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def fixed_point_analysis(r):
    """
    For the logistic map, the fixed points satisfy:
       x* = r * x* (1 - x*)
    
    The fixed points are:
       x* = 0,  and x* = 1 - 1/r   (for r > 1)
    
    Also computes the derivative at the fixed points:
       f'(x) = r (1 - 2x)
    
    Returns:
        fp1, fp2, stab1, stab2: fixed points and booleans for stability (True if |f'(x*)| < 1)
    """
    fp1 = 0.0
    stab1 = abs(r * (1 - 2*fp1)) < 1
    fp2 = None
    stab2 = None
    if r > 1:
        fp2 = 1 - 1/r
        stab2 = abs(r * (1 - 2*fp2)) < 1
    return fp1, fp2, stab1, stab2

def plot_time_series(r, x0, n_iter):
    """
    Plot the time series of the logistic map for a given r.
    """
    x = logistic_map(r, x0, n_iter)
    plt.figure(figsize=(8, 4))
    plt.plot(x, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("x")
    plt.title(f"Time Series of Logistic Map (r = {r})")
    plt.grid(True)
    plt.show()

def plot_bifurcation(r_min, r_max, n_r, n_iter, n_last):
    """
    Plot the bifurcation diagram for the logistic map.
    
    Parameters:
        r_min (float): Minimum r value.
        r_max (float): Maximum r value.
        n_r (int): Number of r values to sample.
        n_iter (int): Total number of iterations for each r value.
        n_last (int): Number of last iterations to plot (to show asymptotic behavior).
    """
    r_values = np.linspace(r_min, r_max, n_r)
    xs = []
    rs = []
    for r in r_values:
        x = logistic_map(r, 0.5, n_iter)
        xs.extend(x[-n_last:])  # only take the last n_last iterations
        rs.extend([r] * n_last)
    plt.figure(figsize=(10, 6))
    plt.plot(rs, xs, ',k', alpha=0.25)
    plt.xlabel("r")
    plt.ylabel("x")
    plt.title("Bifurcation Diagram of the Logistic Map")
    plt.grid(True)
    plt.show()

def lyapunov_exponent(r, x0, n_iter):
    """
    Compute the Lyapunov exponent for the logistic map.
    
    The Lyapunov exponent is given by:
       L = lim_{n->∞} (1/n) \sum_{i=0}^{n-1} ln|f'(x_i)|
    where f'(x) = r*(1-2x).
    
    Parameters:
        r (float): Control parameter.
        x0 (float): Initial condition.
        n_iter (int): Number of iterations.
    
    Returns:
        float: The estimated Lyapunov exponent.
    """
    x = logistic_map(r, x0, n_iter)
    L_sum = 0.0
    for i in range(n_iter - 1):
        derivative = abs(r * (1 - 2 * x[i]))
        # Avoid log(0); if derivative is zero, use a small number
        if derivative == 0:
            derivative = 1e-10
        L_sum += np.log(derivative)
    return L_sum / (n_iter - 1)

def plot_lyapunov(r_min, r_max, n_r, n_iter):
    """
    Plot the Lyapunov exponent as a function of r.
    """
    r_values = np.linspace(r_min, r_max, n_r)
    lyap = np.zeros_like(r_values)
    for i, r in enumerate(r_values):
        lyap[i] = lyapunov_exponent(r, 0.5, n_iter)
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, lyap, 'b-')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("r")
    plt.ylabel("Lyapunov Exponent")
    plt.title("Lyapunov Exponent of the Logistic Map")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example analysis for a fixed r value
    r_value = 3.2
    n_iter = 100
    x0 = 0.5
    print("Logistic Map Analysis for r =", r_value)
    fp1, fp2, stab1, stab2 = fixed_point_analysis(r_value)
    print("Fixed point 1 (x = 0): Stability =", stab1)
    if fp2 is not None:
        print(f"Fixed point 2 (x = {fp2:.4f}): Stability =", stab2)
    else:
        print("No non-zero fixed point exists for r <= 1.")
    
    # Plot time series for a specific r
    plot_time_series(r_value, x0, n_iter)
    
    # Plot bifurcation diagram: use a large number of iterations for convergence
    plot_bifurcation(r_min=2.5, r_max=4.0, n_r=10000, n_iter=1000, n_last=100)
    
    # Plot Lyapunov exponent as a function of r
    plot_lyapunov(r_min=2.5, r_max=4.0, n_r=1000, n_iter=1000)
