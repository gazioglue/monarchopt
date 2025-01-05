"""
Example of solving Multiple Knapsack Problem (MKP) using MSO.
This example shows how to use problem file loading and custom fitness calculation.
"""

from monarchopt import MSO
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MKPData:
    """Class to hold MKP problem data."""
    n_items: int             # Number of items
    n_knapsacks: int         # Number of knapsacks
    profits: np.ndarray      # Profit of each item
    weights: np.ndarray      # Weight matrix [n_items x n_knapsacks]
    capacities: np.ndarray   # Capacity of each knapsack
    known_optimum: float     # Known optimal solution value
    dim: int                 # Problem dimension (same as n_items)

def read_mkp_file(filepath: str = "mkp_instance.txt") -> Tuple[MKPData, float]:
    """Read MKP instance from file."""
    with open(filepath, 'r') as f:
        # Read problem dimensions
        n_items, n_knapsacks = map(int, f.readline().split())
        
        # Read profits
        profits = np.array(list(map(float, f.readline().split())))
        
        # Read weights
        weights = np.zeros((n_items, n_knapsacks))
        for i in range(n_knapsacks):
            weights[:, i] = list(map(float, f.readline().split()))
            
        # Read capacities
        capacities = np.array(list(map(float, f.readline().split())))
        
        # Read known optimum if available
        known_optimum = float(f.readline())
        
        mkp_data = MKPData(
            n_items=n_items,
            n_knapsacks=n_knapsacks,
            profits=profits,
            weights=weights,
            capacities=capacities,
            known_optimum=known_optimum,
            dim=n_items
        )
        
        return mkp_data, known_optimum

def calculate_mkp_fitness(solution: np.ndarray, data: MKPData) -> float:
    """Calculate fitness for MKP solution."""
    # Calculate total profit
    total_profit = np.sum(solution * data.profits)
    
    # Check capacity constraints
    weights_sum = np.dot(solution, data.weights)
    if np.any(weights_sum > data.capacities):
        return -np.sum(np.maximum(0, weights_sum - data.capacities)) * 1000
        
    return total_profit

def main():
    # Run optimization using MSO
    MSO.run(
        obj_func=calculate_mkp_fitness,    # Our fitness function
        dim=None,                          # Will be set from problem file
        pop_size=100,                      # Population size
        max_iter=500,                      # Maximum iterations
        obj_type='max',                    # Maximize the objective
        neighbour_count=5,                 # Number of neighbors
        gradient_strength=0.8,             # Gradient strength
        base_learning_rate=0.1,           # Learning rate
        load_problem_file=read_mkp_file,  # Problem file loader
        known_optimum=20.0,               # Known optimum value
        tolerance=1e-6,                   # Tolerance for convergence
        timeout=3600                      # 1 hour timeout
    )
    
if __name__ == "__main__":
    main()