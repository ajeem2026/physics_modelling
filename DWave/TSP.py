# D-Wave Ising Model Quantum Annealing Project (Starter Notebook)

# ---- TSP: 4-City Traveling Salesman Problem ----

import numpy as np
import itertools
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

# Step 1: Define 4-city distance matrix
D = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

N = 4  # number of cities
A = 100  # penalty weight

# Step 2: Build QUBO dictionary
Q = {}

# Objective: Minimize total distance
for i in range(N):
    for j in range(N):
        for k in range(N):
            if i != k:
                for t in range(N):
                    t_next = (t + 1) % N
                    Q[((i, t), (k, t_next))] = Q.get(((i, t), (k, t_next)), 0) + D[i][k] / 2

# Constraint 1: Each city appears once
for i in range(N):
    for t in range(N):
        Q[((i, t), (i, t))] = Q.get(((i, t), (i, t)), 0) - 2 * A
    for t1 in range(N):
        for t2 in range(t1 + 1, N):
            Q[((i, t1), (i, t2))] = Q.get(((i, t1), (i, t2)), 0) + 2 * A

# Constraint 2: Each position has one city
for t in range(N):
    for i in range(N):
        Q[((i, t), (i, t))] = Q.get(((i, t), (i, t)), 0) - 2 * A
    for i1 in range(N):
        for i2 in range(i1 + 1, N):
            Q[((i1, t), (i2, t))] = Q.get(((i1, t), (i2, t)), 0) + 2 * A

# Convert variable labels to strings for compatibility with D-Wave
Q_labeled = {(f"x_{i}_{t}", f"x_{j}_{s}"): val for ((i, t), (j, s)), val in Q.items()}

# Step 3: Build and sample BQM
bqm = BinaryQuadraticModel.from_qubo(Q_labeled)
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample(bqm, num_reads=100)

# Step 4: Analyze results
print("Best route and energy:")
for sample, energy in response.data(['sample', 'energy']):
    route = sorted([(int(var.split('_')[1]), int(var.split('_')[2])) for var, val in sample.items() if val == 1], key=lambda x: x[1])
    path = [i for i, _ in route]
    print("Route:", path, "Energy:", energy)
    break

# Optionally visualize most frequent routes or compute tour distances
