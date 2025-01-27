# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan 16 13:18:52 2025

# @author: mazilui+ChatGPT
# """

# # Python Code: Oscillations Analysis
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# # Simple Harmonic Oscillator Function
# def simple_harmonic_oscillator(y, t, m, k):
#     """
#     Defines the system of ODEs for a simple harmonic oscillator.
#     """
#     x, v = y
#     dydt = [v, -k / m * x]
#     return dydt

# # Damped Harmonic Oscillator Function
# def damped_oscillator(y, t, m, k, b):
#     """
#     Defines the system of ODEs for a damped harmonic oscillator.
#     """
#     x, v = y
#     dydt = [v, -b / m * v - k / m * x]
#     return dydt

# # # Driven Damped Harmonic Oscillator Function
# # def driven_damped_oscillator(y, t, m, k, b, F, omega):
# #     """
# #     Defines the system of ODEs for a driven damped harmonic oscillator.
# #     """
# #     x, v = y
# #     dydt = [v, -b / m * v - k / m * x + F / m * np.cos(omega * t)]
# #     return dydt

# # # Parameters
# # m = 1.0  # Mass (kg)
# # k = 10.0  # Spring constant (N/m)
# # x0 = 1.0  # Initial position (m)
# # v0 = 0.0  # Initial velocity (m/s)
# # t_end = 20.0  # Simulation time (s)
# # dt = 0.01  # Time step (s)
# # t = np.arange(0, t_end, dt)

# # # Case 1: No Damping, No External Force
# # y0 = [x0, v0]
# # solution = odeint(simple_harmonic_oscillator, y0, t, args=(m, k))
# # x = solution[:, 0]
# # plt.figure(figsize=(10, 6))
# # plt.plot(t, x, label='No Damping, No External Force')
# # plt.title('No Damping, No External Force')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Position x(t)')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # # Case 2: Damped Oscillator
# # b_values = [1.0, 2.0, 10.0]  # Underdamped, Critically damped, Overdamped
# # labels = ['Underdamped', 'Critically Damped', 'Overdamped']
# # plt.figure(figsize=(10, 6))
# # for b, label in zip(b_values, labels):
# #     solution = odeint(damped_oscillator, y0, t, args=(m, k, b))
# #     x = solution[:, 0]
# #     plt.plot(t, x, label=f'{label}: b={b}')
# # plt.title('Damped Oscillator: Underdamped, Critically Damped, Overdamped')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Position x(t)')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # # Case 3: Driven Damped Oscillator
# # b = 1.0  # Damping coefficient (kg/s)
# # F = 5.0  # Driving force amplitude (N)
# # omega = 1.5  # Driving angular frequency (rad/s) (chosen to show transient behavior)
# # solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
# # x = solution[:, 0]
# # plt.figure(figsize=(10, 6))
# # plt.plot(t, x, label=f'Driven Damped: F={F}, omega={omega}')
# # plt.title('Driven Damped Oscillator with Transient Behavior')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Position x(t)')
# # plt.legend()
# # plt.grid()
# # plt.show()

# # Driven Damped Harmonic Oscillator Function
# def driven_damped_oscillator(y, t, m, k, b, F, omega):
#     """
#     Defines the system of ODEs for a driven damped harmonic oscillator.
#     """
#     x, v = y
#     dydt = [v, -b / m * v - k / m * x + F / m * np.cos(omega * t)]
#     return dydt

# # Parameters
# m = 1.0  # Mass (kg)
# k = 10.0  # Spring constant (N/m)
# x0 = 1.0  # Initial position (m)
# v0 = 0.0  # Initial velocity (m/s)
# t_end = 50.0  # Extended simulation time to show steady-state (s)
# dt = 0.01  # Time step (s)
# t = np.arange(0, t_end, dt)
# F = 5.0  # Driving force amplitude (N)
# y0 = [x0, v0]  # Initial conditions

# # Resonance frequency
# omega_res = np.sqrt(k / m)  # Natural frequency of the system
# print(f"Resonance frequency: omega_res = {omega_res:.2f} rad/s")

# # Case 1: Varying Driving Frequencies (ω)
# omega_values = [1.0, 2.5, omega_res, 5.0]  # Driving angular frequencies
# b = 1.0  # Fixed damping coefficient
# plt.figure(figsize=(10, 6))
# for omega in omega_values:
#     solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
#     x = solution[:, 0]
#     plt.plot(t, x, label=f'ω={omega:.2f}')
# plt.title('Driven Damped Oscillator: Varying Driving Frequencies (b=1.0)')
# plt.xlabel('Time (s)')
# plt.ylabel('Position x(t)')
# plt.axvline(t[-1] * 0.2, color='gray', linestyle='--', alpha=0.7, label='Transient/Steady Split')
# plt.legend()
# plt.grid()
# plt.show()

# # Case 2: Varying Damping Coefficients (b)
# b_values = [0.5, 1.0, 5.0]  # Different damping coefficients
# omega = omega_res  # Fixed near-resonance frequency
# plt.figure(figsize=(10, 6))
# for b in b_values:
#     solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
#     x = solution[:, 0]
#     plt.plot(t, x, label=f'b={b}')
# plt.title('Driven Damped Oscillator: Varying Damping Coefficients (ω=ω_res)')
# plt.xlabel('Time (s)')
# plt.ylabel('Position x(t)')
# plt.legend()
# plt.grid()
# plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:18:52 2025

@author: mazilui+ChatGPT
"""

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Driven Damped Harmonic Oscillator Function
def driven_damped_oscillator(y, t, m, k, b, F, omega):
    """
    Defines the system of ODEs for a driven damped harmonic oscillator.
    Args:
        y: Array of [x, v], where x is position and v is velocity.
        t: Time array (s)
        m: Mass (kg)
        k: Spring constant (N/m)
        b: Damping coefficient (kg/s)
        F: Driving force amplitude (N)
        omega: Driving angular frequency (rad/s)
    Returns:
        dydt: List containing [dx/dt, dv/dt].
    """
    x, v = y  # Unpack position (x) and velocity (v)
    dydt = [v, -b / m * v - k / m * x + F / m * np.cos(omega * t)]
    return dydt

# Parameters for the system
m = 1.0        # Mass (kg)
k = 10.0       # Spring constant (N/m)
x0 = 1.0       # Initial position (m)
v0 = 0.0       # Initial velocity (m/s)
F = 5.0        # Driving force amplitude (N)
t_end = 50.0   # Simulation time (s)
dt = 0.01      # Time step (s)
t = np.arange(0, t_end, dt)  # Time array
y0 = [x0, v0]  # Initial conditions: [position, velocity]

# Calculate resonance frequency
omega_res = np.sqrt(k / m)  # Resonance frequency (rad/s)
print(f"Resonance frequency: omega_res = {omega_res:.2f} rad/s")

# ---------------------------------------
# Case 1: Varying Driving Frequencies (ω)
# ---------------------------------------
omega_values = [1.0, 2.5, omega_res, 5.0]  # Driving angular frequencies
b = 1.0  # Fixed damping coefficient
plt.figure(figsize=(10, 6))

# Loop over driving frequencies
for omega in omega_values:
    # Solve the ODE for the current driving frequency
    solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
    x = solution[:, 0]  # Extract position x(t)
    plt.plot(t, x, label=f'ω={omega:.2f}')

# Highlight transient and steady-state regions
transient_cutoff = int(len(t) * 0.2)  # First 20% of the time as transient
plt.axvspan(t[0], t[transient_cutoff], color='red', alpha=0.2, label='Transient Region')
plt.axvspan(t[transient_cutoff], t[-1], color='blue', alpha=0.1, label='Steady-State Region')

# Finalize plot
plt.title('Driven Damped Oscillator: Varying Driving Frequencies (b=1.0)')
plt.xlabel('Time (s)')
plt.ylabel('Position x(t)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------
# Case 2: Varying Damping Coefficients (b)
# ---------------------------------------
b_values = [0.5, 1.0, 5.0]  # Different damping coefficients
omega = omega_res  # Fixed driving frequency near resonance
plt.figure(figsize=(10, 6))

# Loop over damping coefficients
for b in b_values:
    # Solve the ODE for the current damping coefficient
    solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
    x = solution[:, 0]  # Extract position x(t)
    plt.plot(t, x, label=f'b={b}')

# Highlight transient and steady-state regions
plt.axvspan(t[0], t[transient_cutoff], color='red', alpha=0.2, label='Transient Region')
plt.axvspan(t[transient_cutoff], t[-1], color='blue', alpha=0.1, label='Steady-State Region')

# Finalize plot
plt.title('Driven Damped Oscillator: Varying Damping Coefficients (ω=ω_res)')
plt.xlabel('Time (s)')
plt.ylabel('Position x(t)')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------
# Additional Analysis: Steady-State Amplitude vs. ω
# ---------------------------------------
steady_state_amplitudes = []  # Store steady-state amplitudes
b = 1.0  # Fixed damping coefficient
plt.figure(figsize=(8, 5))

# Loop over driving frequencies to calculate steady-state amplitude
for omega in omega_values:
    solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
    x = solution[:, 0]  # Extract position x(t)

    # Extract steady-state region and compute amplitude
    steady_state_x = x[transient_cutoff:]  # Steady-state region (after transient)
    steady_state_amplitude = np.max(steady_state_x) - np.min(steady_state_x)
    steady_state_amplitudes.append(steady_state_amplitude)

# Plot steady-state amplitude vs. ω
plt.plot(omega_values, steady_state_amplitudes, marker='o')
plt.title('Steady-State Amplitude vs. Driving Frequency (b=1.0)')
plt.xlabel('Driving Frequency ω (rad/s)')
plt.ylabel('Steady-State Amplitude')
plt.grid()
plt.show()

# ---------------------------------------
# Additional Analysis: Steady-State Amplitude vs. b
# ---------------------------------------
steady_state_amplitudes_b = []  # Store steady-state amplitudes
omega = omega_res  # Fixed driving frequency near resonance
plt.figure(figsize=(8, 5))

# Loop over damping coefficients to calculate steady-state amplitude
for b in b_values:
    solution = odeint(driven_damped_oscillator, y0, t, args=(m, k, b, F, omega))
    x = solution[:, 0]  # Extract position x(t)

    # Extract steady-state region and compute amplitude
    steady_state_x = x[transient_cutoff:]  # Steady-state region (after transient)
    steady_state_amplitude = np.max(steady_state_x) - np.min(steady_state_x)
    steady_state_amplitudes_b.append(steady_state_amplitude)

# Plot steady-state amplitude vs. b
plt.plot(b_values, steady_state_amplitudes_b, marker='o')
plt.title('Steady-State Amplitude vs. Damping Coefficient (ω=ω_res)')
plt.xlabel('Damping Coefficient b (kg/s)')
plt.ylabel('Steady-State Amplitude')
plt.grid()
plt.show()
