#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:08:49 2025

@author: mazilui+ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt

# Road parameters
L = 100          # Length of the road
density = 0.3    # Car density (fraction of cells occupied)
N_cars = int(L * density)
road = -1 * np.ones(L, dtype=int)  # -1 indicates an empty cell

# Randomly assign car positions; initial speed is 0 for all cars
positions = np.random.choice(range(L), N_cars, replace=False)
road[positions] = 0

def update(road, max_speed=5, p_slow=0.3):
    L = len(road)
    new_road = -1 * np.ones(L, dtype=int)
    for i in range(L):
        if road[i] != -1:
            speed = road[i]
            # Acceleration: increase speed up to max_speed
            speed = min(speed + 1, max_speed)
            # Gap checking: determine distance to the next car
            gap = 1
            while road[(i + gap) % L] == -1 and gap <= speed:
                gap += 1
            speed = min(speed, gap - 1)
            # Random slowing: simulate driver behavior and perturbations
            if np.random.rand() < p_slow and speed > 0:
                speed -= 1
            new_pos = (i + speed) % L
            new_road[new_pos] = speed
    return new_road

# Run the simulation and collect states for visualization
timesteps = 50
states = [road.copy()]
for t in range(timesteps):
    road = update(road)
    states.append(road.copy())

plt.imshow(np.array(states), cmap='viridis', aspect='auto')
plt.title("Traffic Flow Simulation: Formation of Jams")
plt.xlabel("Position")
plt.ylabel("Time Step")
plt.show()
