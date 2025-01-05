"""
Basic usage example of the MSO algorithm.
This example demonstrates solving a simple binary optimization problem.
"""

from monarchopt import MSO
import numpy as np

def simple_fitness(solution):
    """Simple fitness function: maximize the number of 1s while keeping first bit 0."""
    if solution[0] == 1:  # Constraint: first bit must be 0
        return -1000
    return np.sum(solution[1:])

def main():
    # Run optimization
    MSO.run(
        obj_func=simple_fitness, # Our fitness function
        dim=20,                  # 20-bit problem
        pop_size=50,             # Population size
        max_iter=100,            # Maximum iterations
        obj_type='max',          # Maximize the objective
        neighbour_count=3,       # Number of neighbors to interact with
        gradient_strength=0.8,   # Default gradient strength
        base_learning_rate=0.1,  # Default learning rate
        timeout=60               # 60 seconds timeout
    )
    
if __name__ == "__main__":
    main()