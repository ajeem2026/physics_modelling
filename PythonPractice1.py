"""
@author Abid Jeem, Jonathan Carranza Cortes, ChatGPT

"""


import numpy as np
import matplotlib.pyplot as plt

#========================================================
# Kinetic Energy Problem
def calculate_kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy of an object.

    Parameters:
    mass (float): The mass of the object (in kilograms).
    velocity (float): The velocity of the object (in meters per second).

    Returns:
    float: The kinetic energy (in joules).
    """
    return 0.5 * mass * velocity ** 2

# Prompt user for input
try:
    mass = float(input("Enter the mass of the object (in kilograms): "))
    velocity = float(input("Enter the velocity of the object (in meters per second): "))

    if mass < 0 or velocity < 0:
        print("Mass and velocity must be non-negative values.")
    else:
        # Calculate kinetic energy
        kinetic_energy = calculate_kinetic_energy(mass, velocity)
        print(f"The kinetic energy of the object is {kinetic_energy:.2f} joules.")
except ValueError:
    print("Please enter valid numerical values for mass and velocity.")


#========================================================
# Free Fall Problem
# Program to calculate the distance an object falls due to gravity over time

def calculate_falling_distance(time, gravity=9.8):
    """
    Calculate the distance an object falls due to gravity.

    Parameters:
    time (float): The time the object has been falling (in seconds).
    gravity (float): The acceleration due to gravity (in m/s^2). Default is 9.8.

    Returns:
    float: The distance fallen (in meters).
    """
    return 0.5 * gravity * time ** 2

# Loop through times from 1 to 10 seconds
print("Time (s)\tDistance Fallen (m)")
print("---------------------------")
for time in range(1, 11):
    distance = calculate_falling_distance(time)
    print(f"{time}\t\t{distance:.2f}")
    
#========================================================
# Ohm's Law Problem
# Function to calculate the current in a circuit
def calculate_current(voltage, resistance):
    """
    Calculate the current in a circuit.

    Parameters:
    voltage (float): The voltage across the circuit (in volts).
    resistance (float): The resistance in the circuit (in ohms).

    Returns:
    float: The current (in amperes).
    """
    if resistance <= 0:
        raise ValueError("Resistance must be a positive value.")
    return voltage / resistance

# Test the function with V = 12 V and R = 6 Ω
try:
    voltage = 12  # in volts
    resistance = 6  # in ohms

    current = calculate_current(voltage, resistance)
    print(f"The current in the circuit is {current:.2f} amperes.")
except ValueError as e:
    print(e)
    
    
#========================================================
# Projectile Motion Problem

# Constants
g = 9.8  # acceleration due to gravity in m/s^2
v0 = 20  # initial velocity in m/s

# Function to calculate the horizontal range of a projectile
def calculate_range(initial_velocity, angle_degrees, gravity):
    """
    Calculate the horizontal range of a projectile.

    Parameters:
    initial_velocity (float): The initial velocity of the projectile (in m/s).
    angle_degrees (float): The launch angle (in degrees).
    gravity (float): The acceleration due to gravity (in m/s^2).

    Returns:
    float: The horizontal range (in meters).
    """
    angle_radians = np.radians(angle_degrees)
    return (initial_velocity ** 2) * np.sin(2 * angle_radians) / gravity

# Calculate and display ranges for angles from 10° to 90° in steps of 10°
angles = np.arange(10, 91, 10)
print("Angle (°)\tRange (m)")
print("-----------------------")

for angle in angles:
    range_distance = calculate_range(v0, angle, g)
    print(f"{angle}\t\t{range_distance:.2f}")
