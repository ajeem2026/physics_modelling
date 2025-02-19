import numpy as np
import matplotlib.pyplot as plt

# Define physical parameters
d = 1.0            # Plate separation (in meters)
V0 = 1.0           # Potential at the top plate (in volts)
rho0 = 1e-6        # Uniform charge density (in coulombs per cubic meter)
epsilon = 8.85e-12 # Permittivity of free space (in F/m)

# Define grid resolution for numerical computation
ny, nx = 100, 100  # Number of grid points in the y and x directions
y = np.linspace(0, d, ny)  # Create an array of y-coordinates from 0 to d
x = np.linspace(0, d, nx)  # Create an array of x-coordinates from 0 to d

# ------------------------------------------------------------------------------
# Analytical Solution (1D along y)
# ------------------------------------------------------------------------------
# We solve the 1D Poisson equation: d^2(phi)/dy^2 = -rho0/epsilon
# with the boundary conditions:
#   phi(0) = 0      (bottom plate at 0 V)
#   phi(d) = V0     (top plate at V0)
# The general solution for this second order differential equation is:
#   phi(y) = - (rho0/(2*epsilon)) * y^2 + A * y + B
# Applying the boundary conditions, we get B = 0 and A such that phi(d)=V0.
phi_analytical = - (rho0 / (2 * epsilon)) * y**2 + ((V0 + (rho0 / (2 * epsilon)) * d**2) / d) * y

# Compute the analytical electric field.
# The electric field E is the negative gradient of the potential.
E_analytical = np.gradient(-phi_analytical, y)

# ------------------------------------------------------------------------------
# Numerical Solution using Jacobi Method (2D version)
# ------------------------------------------------------------------------------
def solve_poisson_jacobi(ny, nx, V0, rho0, epsilon, max_iter=50000, tol=1e-8):
    """
    Solve the 2D Poisson equation on a square domain using the Jacobi method.
    
    Domain:
      - The domain is a square of side length d.
      - The bottom boundary (y=0) is fixed at 0 V.
      - The top boundary (y=d) is fixed at V0.
      - The left and right boundaries use Neumann (zero-flux) boundary conditions.
    
    Poisson equation:
      ∇²φ = -ρ/ε, where ρ is the charge density.
      
    Parameters:
      ny, nx    : Number of grid points in the y and x directions.
      V0        : Potential at the top boundary.
      rho0      : Uniform charge density.
      epsilon   : Permittivity of free space.
      max_iter  : Maximum number of iterations for the Jacobi method.
      tol       : Convergence tolerance for stopping the iteration.
      
    Returns:
      phi       : 2D array of the computed potential values.
    """
    # Calculate grid spacing (assume dx = dy for a square grid)
    dy = d / (ny - 1)
    
    # Initialize potential array with zeros; shape is (ny, nx)
    phi = np.zeros((ny, nx))
    
    # Set the top boundary (y=d) to V0
    phi[-1, :] = V0
    
    # The source term f is constant, derived from Poisson's equation:
    #   ∇²φ = -ρ/ε   =>   f = -ρ0/ε
    f = -rho0 / epsilon

    # Start the iterative Jacobi method
    for iteration in range(max_iter):
        # Copy the current potential to update all points simultaneously
        phi_new = phi.copy()
        
        # Update interior points using the Jacobi formula:
        # The new value at each grid point is the average of its four neighbors,
        # adjusted by the source term.
        # We update all points except the boundaries.
        phi_new[1:-1, 1:-1] = 0.25 * (
            phi[2:, 1:-1] +   # potential from the cell below
            phi[:-2, 1:-1] +  # potential from the cell above
            phi[1:-1, 2:] +   # potential from the cell to the right
            phi[1:-1, :-2] -  # potential from the cell to the left
            dy**2 * f        # adjustment due to the source term
        )
        
        # Apply Neumann (zero-flux) boundary conditions at the left and right edges:
        # This means the derivative (flux) normal to the boundary is zero.
        # We set the boundary values equal to the adjacent interior values.
        phi_new[:, 0] = phi_new[:, 1]   # Left boundary
        phi_new[:, -1] = phi_new[:, -2] # Right boundary

        # Calculate the maximum change between the current and updated potential.
        # This is used to check for convergence.
        diff = np.max(np.abs(phi_new - phi))
        
        # Update phi for the next iteration
        phi = phi_new
        
        # If the maximum change is less than the tolerance, the solution has converged.
        if diff < tol:
            print(f'Converged after {iteration+1} iterations')
            break
    else:
        print('Warning: Maximum iterations reached without full convergence')
    
    return phi

# Solve the Poisson equation numerically using the Jacobi method
phi_numerical = solve_poisson_jacobi(ny, nx, V0, rho0, epsilon)

# Compute the numerical electric field.
# We use np.gradient to compute the derivative along the y-axis.
# The second argument, dy, ensures that the spacing between grid points is correctly accounted for.
dy = d / (ny - 1)
E_numerical = np.gradient(-phi_numerical, dy, axis=0)

# ------------------------------------------------------------------------------
# Plotting the Results
# ------------------------------------------------------------------------------
# Create a figure with three subplots to compare the analytical and numerical results.
plt.figure(figsize=(12, 6))

# 1. Plot the potential distribution (comparison along the central x-column)
plt.subplot(1, 3, 1)
plt.plot(y, phi_analytical, label='Analytical Solution', linestyle='dashed', color='red')
# Use the potential from the middle of the grid (x = d/2)
plt.plot(y, phi_numerical[:, nx//2], label='Numerical (Jacobi)', linestyle='solid', color='blue')
plt.xlabel('y-position (m)')
plt.ylabel('Potential (V)')
plt.title('Potential Distribution')
plt.legend()
plt.grid()

# 2. Plot the electric field distribution (comparison along the central x-column)
plt.subplot(1, 3, 2)
plt.plot(y, E_analytical, label='Analytical Electric Field', linestyle='dashed', color='red')
plt.plot(y, E_numerical[:, nx//2], label='Numerical Electric Field', linestyle='solid', color='blue')
plt.xlabel('y-position (m)')
plt.ylabel('Electric Field (V/m)')
plt.title('Electric Field Distribution')
plt.legend()
plt.grid()

# 3. Create a heatmap of the 2D numerical potential distribution
plt.subplot(1, 3, 3)
# imshow displays the 2D array as an image.
# 'origin' is set to 'lower' to display the bottom of the grid at the bottom.
# 'extent' defines the axis scales from 0 to d.
plt.imshow(phi_numerical, cmap='inferno', origin='lower', extent=[0, d, 0, d])
plt.colorbar(label='Potential (V)')
plt.title('Heatmap of Potential Distribution')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')

# Adjust the layout for better spacing between subplots
plt.tight_layout()
plt.show()
