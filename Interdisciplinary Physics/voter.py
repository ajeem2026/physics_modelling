#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:04:46 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt

def voter_model_simulation(N=100, timesteps=1000):
    # Initialize each agent with a random opinion (0 or 1)
    opinions = np.random.choice([0, 1], size=N)
    avg_opinions = []
    
    for t in range(timesteps):
        avg_opinions.append(np.mean(opinions))
        # Choose a random agent
        i = np.random.randint(N)
        # For a ring topology, choose one of the two neighbors at random
        neighbor = np.random.choice([(i-1) % N, (i+1) % N])
        # Update the agent's opinion to that of its neighbor
        opinions[i] = opinions[neighbor]
    
    return avg_opinions, opinions

if __name__ == '__main__':
    avg_opinions, final_state = voter_model_simulation()
    plt.plot(avg_opinions)
    plt.xlabel('Time Step')
    plt.ylabel('Average Opinion')
    plt.title('Voter Model Simulation: Convergence to Consensus')
    plt.show()
