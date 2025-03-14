import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 50              # Lattice size (L x L)
n_sweeps = 5000     # Total number of Monte Carlo sweeps
temperatures = [1.5, 2.5, 3.5]  # Different temperatures to study phase transition
J = 1.0             # Coupling constant
H = 0.0             # External field

def metropolis_sweep(spins, T):
    """ Perform one Monte Carlo sweep using the Metropolis algorithm. """
    for _ in range(L * L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        s = spins[i, j]
        # Sum of nearest neighbors with periodic boundary conditions
        nb = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
        dE = 2 * s * (J * nb + H)
        if dE <= 0 or np.random.rand() < np.exp(-dE/T):
            spins[i, j] = -s
    return spins

# Run the simulation for different temperatures
magnetization_results = {}

for T in temperatures:
    spins = np.random.choice([-1, 1], size=(L, L))  # Initialize spins randomly
    magnetization = []
    for _ in range(n_sweeps):
        spins = metropolis_sweep(spins, T)
        magnetization.append(np.mean(spins))
    magnetization_results[T] = magnetization

# Plot Magnetization per Spin for different temperatures
plt.figure(figsize=(8, 5))
for T, mag in magnetization_results.items():
    plt.plot(range(n_sweeps), mag, label=f'T = {T}')
plt.xlabel('Monte Carlo Sweeps')
plt.ylabel('Magnetization per Spin')
plt.title('Magnetization Evolution at Different Temperatures')
plt.legend()
plt.show()
