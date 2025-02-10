#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:55:18 2025

@author: mazilui+ChatGPT
"""

import sympy as sp

# Define symbols
x, t = sp.symbols('x t')

# 1. Simplification Example:
# Define an expression that should simplify to 1 (since sin^2(x) + cos^2(x) = 1)
expr1 = sp.sin(x)**2 + sp.cos(x)**2
simplified_expr1 = sp.simplify(expr1)
print("Simplified Expression (sin^2(x) + cos^2(x)):", simplified_expr1)

# 2. Differentiation Example:
# Define an expression and differentiate it with respect to x
expr2 = sp.exp(x) * sp.sin(x)
derivative_expr2 = sp.diff(expr2, x)
print("\nExpression: exp(x)*sin(x)")
print("Derivative:", derivative_expr2)

# 3. Integration Example:
# Integrate the expression exp(x)*sin(x) with respect to x
integral_expr2 = sp.integrate(expr2, x)
print("\nIntegral of exp(x)*sin(x):", integral_expr2)

# 4. Solving an Equation:
# Solve the equation exp(x)*sin(x) = 0 for x
# Note: This example will solve symbolically and may return a general solution.
solution = sp.solve(sp.Eq(expr2, 0), x)
print("\nSolutions for exp(x)*sin(x) = 0:")
print(solution)
