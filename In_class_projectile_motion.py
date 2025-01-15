"""
@author Abid Jeem, Jonathan Carranza Cortes, ChatGPT
Assumptions
- Gravity is constant
- No air resistance
- Idealized launch meaning it launches exactly at 45 degrees
- Ground is perfectly flat
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.8  # gravity in m/s^2
v0 = 20  # initial velocity in m/s
angle = 45  # launch angle in degrees
h = 5  # launch height in meters
rho = 1.225  # air density in kg/m^3
C_d = 0.47  # drag coefficient (sphere)
A = 0.01  # cross-sectional area in m^2
m = 0.1  # mass in kg
wind_speed = 5  # wind speed in m/s (horizontal)

# Drag constant
k = 0.5 * rho * C_d * A  

# Time step
dt = 0.01  

# Function for simulation with realistic factors
def simulate_projectile(v0, angle, h, air_resistance=True, wind_speed=0):
    theta = np.radians(angle)  # Convert angle to radians
    vx = v0 * np.cos(theta)  # Initial horizontal velocity
    vy = v0 * np.sin(theta)  # Initial vertical velocity

    x, y = [0], [h]  # Initial positions
    vx_current, vy_current = vx + wind_speed, vy  # Include wind in initial velocity
    time = [0]

    while y[-1] >= 0:  # Simulate until projectile hits the ground
        v = np.sqrt(vx_current**2 + vy_current**2)
        
        # Compute accelerations
        if air_resistance:
            ax = -k * vx_current * v / m
            ay = -g - (k * vy_current * v / m)
        else:
            ax, ay = 0, -g
        
        # Update velocities
        vx_current += ax * dt
        vy_current += ay * dt

        # Update positions
        x.append(x[-1] + vx_current * dt)
        y.append(y[-1] + vy_current * dt)
        time.append(time[-1] + dt)
    
    return np.array(x), np.array(y)

# Simulations
x_no_resistance, y_no_resistance = simulate_projectile(v0, angle, 0, air_resistance=False)
x_with_resistance, y_with_resistance = simulate_projectile(v0, angle, h, air_resistance=True, wind_speed=wind_speed)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: No Air Resistance
axs[0].plot(x_no_resistance, y_no_resistance, label="No Air Resistance", color="blue")
axs[0].set_title("Idealized Projectile Motion", fontsize=14)
axs[0].set_xlabel("Horizontal Distance (m)", fontsize=12)
axs[0].set_ylabel("Vertical Distance (m)", fontsize=12)
axs[0].grid(True)
axs[0].legend(fontsize=12)
axs[0].annotate(
    f"Max Range: {x_no_resistance[-1]:.2f} m",
    xy=(x_no_resistance[-1], 0),
    xytext=(x_no_resistance[-1] * 0.8, -5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=10,
)
axs[0].annotate(
    f"Max Height: {np.max(y_no_resistance):.2f} m",
    xy=(x_no_resistance[-1] / 2, np.max(y_no_resistance)),
    xytext=(x_no_resistance[-1] / 3, np.max(y_no_resistance) + 5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=10,
)

# Plot 2: With Air Resistance and Wind
axs[1].plot(x_with_resistance, y_with_resistance, label="With Air Resistance + Launch Height + Wind", color="orange")
axs[1].set_title("Realistic Projectile Motion", fontsize=14)
axs[1].set_xlabel("Horizontal Distance (m)", fontsize=12)
axs[1].set_ylabel("Vertical Distance (m)", fontsize=12)
axs[1].grid(True)
axs[1].legend(fontsize=12)
axs[1].annotate(
    f"Max Range: {x_with_resistance[-1]:.2f} m",
    xy=(x_with_resistance[-1], 0),
    xytext=(x_with_resistance[-1] * 0.8, -5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=10,
)
axs[1].annotate(
    f"Max Height: {np.max(y_with_resistance):.2f} m",
    xy=(x_with_resistance[-1] / 2, np.max(y_with_resistance)),
    xytext=(x_with_resistance[-1] / 3, np.max(y_with_resistance) + 5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=10,
)

# Title for the entire figure
fig.suptitle("Comparison of Projectile Motion: Idealized vs Realistic", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
