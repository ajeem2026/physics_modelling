import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the SIR model for Dengue infection
def sir_model(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parameters from the research paper
N = 1000000  # Total population (1 million for demonstration)
beta = 0.5  # Transmission rate (contact rate * probability of transmission per contact)
gamma = 1/14  # Recovery rate (assuming 14 days infectious period)
I0 = 1000  # Initial infected individuals
R0 = 0  # Initial recovered individuals
S0 = N - I0 - R0  # Initial susceptible population

# Time span for simulation (days)
t_span = (0, 365)  # Simulate for one year
t_eval = np.linspace(0, 365, 365)  # Daily intervals

# Solve the system using Runge-Kutta method
solution = solve_ivp(sir_model, t_span, [S0, I0, R0], args=(beta, gamma, N), t_eval=t_eval)

# Extract results
S, I, R = solution.y

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, S, label="Susceptible", color='blue')
plt.plot(solution.t, I, label="Infected", color='red')
plt.plot(solution.t, R, label="Recovered", color='green')
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.title("SIR Model for Dengue Infection in Bangladesh")
plt.legend()
plt.grid()
plt.show()
