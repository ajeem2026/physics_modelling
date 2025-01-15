"""

@author Abid Jeem, Jonathan Carranza Cortes, ChatGPT

Assumptions with initial model:

- Gravity is constant
- No air resistance
- Idealized launch meaning it launches exactly at 45 degrees
- Ground is perfectly flat

@scenario Lionel Messi kicks a football
Description:

- Messi kicks a standard football with initial velocity.
- Models realistic projectile motion, considering air resistance, wind, and launch height.
- Assumptions include constant gravity, no spin effects, and flat ground.

Constants:

- Gravity: 9.8 m/s^2
- Football mass: 0.43 kg (standard FIFA football)
- Football diameter: 22 cm, cross-sectional area derived from this

"""

import numpy as np
import matplotlib.pyplot as plt

# Constants specific to Messi's kick and football
g = 9.8  # Gravity (m/s^2)
v0 = 30  # Estimated initial velocity from Messi's kick (m/s)
angle = 45  # Launch angle (degrees)
h = 3.5  # Approximate height of Messi's foot during kick (m)
rho = 1.225  # Air density at sea level (kg/m^3)
C_d = 0.2  # Approximate drag coefficient for a spinning football
d = 0.22  # Diameter of the football (m)
A = np.pi * (d / 2)**2  # Cross-sectional area of the football (m^2)
m = 0.43  # Mass of the football (kg)
wind_speed = 20  # Horizontal wind speed (m/s)

# Drag constant calculation
k = 0.5 * rho * C_d * A  # Drag factor based on velocity

# Time step for simulation (small value for precision)
dt = 0.01

# Function for simulating Messi's kick
def simulate_projectile(v0, angle, h, air_resistance=True, wind_speed=0):
    """
    Simulates the motion of a football kicked by Messi, considering air resistance, wind, and launch height.
    
    Parameters:
    - v0: Initial velocity (m/s)
    - angle: Launch angle (degrees)
    - h: Initial launch height (m)
    - air_resistance: Whether to include air resistance (True/False)
    - wind_speed: Horizontal wind speed (m/s)
    
    Returns:
    - x, y: Arrays of horizontal and vertical positions
    """
    theta = np.radians(angle)  # Convert angle to radians
    vx = v0 * np.cos(theta)  # Initial horizontal velocity
    vy = v0 * np.sin(theta)  # Initial vertical velocity

    x, y = [0], [h]  # Initialize position arrays
    vx_current, vy_current = vx + wind_speed, vy  # Adjust initial velocity for wind
    time = [0]  # Time tracker

    # Loop until the football hits the ground (y <= 0)
    while y[-1] >= 0:
        v = np.sqrt(vx_current**2 + vy_current**2)  # Speed magnitude

        # Calculate accelerations due to air resistance and gravity
        if air_resistance:
            ax = -k * vx_current * v / m  # Horizontal drag
            ay = -g - (k * vy_current * v / m)  # Vertical drag and gravity
        else:
            ax, ay = 0, -g  # No air resistance case

        # Update velocities using calculated accelerations
        vx_current += ax * dt
        vy_current += ay * dt

        # Update positions using updated velocities
        x.append(x[-1] + vx_current * dt)
        y.append(y[-1] + vy_current * dt)
        time.append(time[-1] + dt)

    return np.array(x), np.array(y)

# Simulations
x_no_resistance, y_no_resistance = simulate_projectile(v0, angle, 0, air_resistance=False)
x_with_resistance, y_with_resistance = simulate_projectile(v0, angle, h, air_resistance=True, wind_speed=wind_speed)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Idealized motion (No Air Resistance)
axs[0].plot(x_no_resistance, y_no_resistance, label="No Air Resistance", color="blue")
axs[0].set_title("Idealized Football Kick", fontsize=14)
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

# Plot 2: Realistic motion (With Air Resistance and Wind)
axs[1].plot(x_with_resistance, y_with_resistance, label="With Air Resistance + Launch Height + Wind", color="orange")
axs[1].set_title("Realistic Football Kick", fontsize=14)
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
fig.suptitle("Comparison of Messi's Football Kick: Idealized vs Realistic", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
