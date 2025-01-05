#!/usr/bin/env python3
"""
UFLP (Uncapacitated Facility Location Problem) Solver using Monarch Swarm Optimization.
Usage: python solve_uflp.py <instance_file> [options]

Example:
    python solve_uflp.py cap71.txt
    python solve_uflp.py cap71.txt --pop-size 2000 --max-iter 1000
"""

import argparse
import numpy as np
from pathlib import Path
from monarchopt import MSO
from typing import Tuple, Optional, Dict

# Known optimal values for standard instances
UFLP_OPTIMUMS = {
    'cap71': 932615.750,
    'cap72': 977799.400,
    'cap73': 1010641.450,
    'cap74': 1034976.975,
    'cap101': 796648.437,
    'cap102': 854704.200,
    'cap103': 893782.112,
    'cap104': 928941.750,
    'cap131': 793439.562,
    'cap132': 851495.325,
    'cap133': 893076.712,
    'cap134': 928941.750,
    'capa': 17156454.478,
    'capb': 12979071.582,
    'capc': 11505594.329
}

def read_uflp_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, int, int, Optional[float]]:
    """Read UFLP instance file.
    
    Returns:
        Tuple containing:
        - costs: Facility opening costs
        - distances: Customer-facility distances
        - n_facilities: Number of facilities
        - n_customers: Number of customers
        - known_optimum: Known optimal value (if available)
    """
    with open(filepath, 'r') as f:
        # Read dimensions
        n_facilities, n_customers = map(int, f.readline().strip().split())
        
        # Read facility costs
        costs = np.zeros(n_facilities)
        for i in range(n_facilities):
            _, cost = map(float, f.readline().strip().split())
            costs[i] = cost
        
        # Read distances
        distances = np.zeros((n_customers, n_facilities))
        buffer = []
        for i in range(n_customers):
            while len(buffer) < n_facilities + 1:
                buffer.extend(f.readline().strip().split())
            values = list(map(float, buffer[:n_facilities + 1]))
            buffer = buffer[n_facilities + 1:]
            distances[i] = values[1:]
    
    # Get known optimum if available
    instance_name = Path(filepath).stem
    known_optimum = UFLP_OPTIMUMS.get(instance_name)
    
    return costs, distances, n_facilities, n_customers, known_optimum

def calculate_uflp_fitness(solution: np.ndarray, costs: np.ndarray, distances: np.ndarray) -> float:
    """Calculate total cost for a UFLP solution."""
    try:
        if solution is None or len(solution) == 0:
            return float('inf')
        
        facility_count = np.sum(solution)
        if facility_count == 0:
            return float('inf')
        
        # Calculate facility opening costs
        fixed_cost = np.sum(solution * costs)
        
        # Calculate transportation costs
        transport_cost = 0
        distances_t = distances.T
        
        for j in range(distances.shape[0]):  # For each customer
            costs_to_customer = distances_t[:, j]
            open_facilities = np.where(solution == 1)[0]
            if len(open_facilities) == 0:
                return float('inf')
            
            min_cost = min(costs_to_customer[open_facilities])
            transport_cost += min_cost
        
        return fixed_cost + transport_cost
        
    except Exception as e:
        print(f"Error in UFLP calculation: {str(e)}")
        return float('inf')

def main():
    parser = argparse.ArgumentParser(description='Solve UFLP using Monarch Swarm Optimization')
    parser.add_argument('instance_file', type=str, help='Path to UFLP instance file')
    parser.add_argument('--pop-size', type=int, default=1000, help='Population size')
    parser.add_argument('--max-iter', type=int, default=800, help='Maximum iterations')
    parser.add_argument('--neighbour-count', type=int, default=3, help='Number of neighbors')
    parser.add_argument('--gradient-strength', type=float, default=0.8, help='Gradient field strength')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save-results', choices=['yes', 'no'], default='yes', help='Save results to file')
    args = parser.parse_args()
    
    # Load problem data
    costs, distances, n_facilities, n_customers, known_optimum = read_uflp_file(args.instance_file)
    print(f"\nProblem dimensions: {n_facilities} facilities, {n_customers} customers")
    if known_optimum:
        print(f"Known optimal value: {known_optimum}")
    
    # Create fitness function closure
    def fitness_func(solution):
        return calculate_uflp_fitness(solution, costs, distances)
    
    # Run optimization
    instance_name = Path(args.instance_file).stem
    MSO.run(
        obj_func=fitness_func,
        dim=n_facilities,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        obj_type='min',
        neighbour_count=args.neighbour_count,
        gradient_strength=args.gradient_strength,
        base_learning_rate=args.learning_rate,
        known_optimum=known_optimum,
        tolerance=args.tolerance,
        seed=args.seed,
        save_results=args.save_results == 'yes',
        results_dir=f"results/uflp/{instance_name}"
    )

if __name__ == "__main__":
    main()