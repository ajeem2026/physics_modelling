#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:06:06 2025

@author: mazilui
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_agents = 200
timesteps = 100
sample_size = 20

# Initialize agents with either buy (+1) or sell (-1) decisions
opinions = np.random.choice([-1, 1], size=N_agents)
avg_opinion = [np.mean(opinions)]

for t in range(timesteps):
    new_opinions = np.empty(N_agents, dtype=int)
    for i in range(N_agents):
        # Each agent samples a subset of agents
        sample = np.random.choice(opinions, size=sample_size, replace=True)
        # Update the agent's opinion based on the sign of the sample mean
        new_opinions[i] = 1 if np.mean(sample) >= 0 else -1
    opinions = new_opinions
    avg_opinion.append(np.mean(opinions))

plt.figure(figsize=(8, 4))
plt.plot(avg_opinion, marker='o', linestyle='-', color='blue')
plt.title("Market Opinion Evolution in an Agent-Based Model")
plt.xlabel("Time Step")
plt.ylabel("Average Opinion")
plt.grid(True)
plt.show()
