#!/usr/bin/env python3
"""
DUF (Decomposable Unitation-based Functions) Solver using Monarch Swarm Optimization.
Usage: python solve_dufs.py <duf_number> [options]

Example:
    python solve_dufs.py duf1
    python solve_dufs.py duf2 --dim 200 --pop-size 200
"""

import argparse
import numpy as np
from monarchopt import MSO

def calculate_block_fitness(block: np.ndarray) -> int:
    """Calculate block fitness (v value)."""
    return np.sum(block)

def DUF1(solution: np.ndarray) -> float:
    """OneMax function: maximize number of ones."""
    return np.sum(solution)

def DUF2(solution: np.ndarray) -> float:
    """Plateau function."""
    if len(solution) % 4 != 0:
        raise ValueError("Solution length must be divisible by 4")
    
    total_fitness = 0
    num_blocks = len(solution) // 4
    
    for i in range(num_blocks):
        block = solution[i*4:(i+1)*4]
        v = calculate_block_fitness(block)
        
        if v == 4:
            sub_fitness = 4
        elif v == 3:
            sub_fitness = 2
        else:
            sub_fitness = 0
            
        total_fitness += sub_fitness
    
    return total_fitness

def DUF3(solution: np.ndarray) -> float:
    """Deceptive function."""
    if len(solution) % 4 != 0:
        raise ValueError("Solution length must be divisible by 4")
    
    total_fitness = 0
    num_blocks = len(solution) // 4
    
    for i in range(num_blocks):
        block = solution[i*4:(i+1)*4]
        v = calculate_block_fitness(block)
        
        if v == 4:
            sub_fitness = 4
        else:
            sub_fitness = 3 - v
            
        total_fitness += sub_fitness
    
    return total_fitness

def DUF4(solution: np.ndarray) -> float:
    """Royal Road function."""
    if len(solution) % 4 != 0:
        raise ValueError("Solution length must be divisible by 4")
    
    total_fitness = 0
    num_blocks = len(solution) // 4
    
    for i in range(num_blocks):
        block = solution[i*4:(i+1)*4]
        v = calculate_block_fitness(block)
        
        sub_fitness = 4 if v == 4 else 0
        total_fitness += sub_fitness
    
    return total_fitness

DUF_FUNCTIONS = {
    'duf1': DUF1,
    'duf2': DUF2,
    'duf3': DUF3,
    'duf4': DUF4
}

def main():
    parser = argparse.ArgumentParser(description='Solve DUF problems using Monarch Swarm Optimization')
    parser.add_argument('duf', type=str, choices=list(DUF_FUNCTIONS.keys()),
                       help='DUF function to solve')
    parser.add_argument('--dim', type=int, default=100,
                       help='Problem dimension (must be divisible by 4 for DUF2-4)')
    parser.add_argument('--pop-size', type=int, default=1000,
                       help='Population size')
    parser.add_argument('--max-iter', type=int, default=800,
                       help='Maximum iterations')
    parser.add_argument('--neighbour-count', type=int, default=3,
                       help='Number of neighbors')
    parser.add_argument('--gradient-strength', type=float, default=0.8,
                       help='Gradient field strength')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save-results', choices=['yes', 'no'], default='yes',
                       help='Save results to file')
    args = parser.parse_args()
    
    # Get selected DUF function
    duf_func = DUF_FUNCTIONS[args.duf]
    print(f"\nSolving {args.duf.upper()} with dimension {args.dim}")
    
    # For DUF2-4, dimension must be divisible by 4
    if args.duf != 'duf1' and args.dim % 4 != 0:
        raise ValueError(f"Dimension must be divisible by 4 for {args.duf}")
    
    # Calculate known optimum if possible
    if args.duf == 'duf1':
        known_optimum = args.dim  # All ones
    else:
        known_optimum = (args.dim // 4) * 4  # Maximum possible value
    
    # Run optimization
    MSO.run(
        obj_func=duf_func,
        dim=args.dim,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        obj_type='max',
        neighbour_count=args.neighbour_count,
        gradient_strength=args.gradient_strength,
        base_learning_rate=args.learning_rate,
        known_optimum=known_optimum,
        tolerance=1e-6,
        seed=args.seed,
        save_results=args.save_results == 'yes',
        results_dir=f"results/{args.duf}_d{args.dim}"
    )

if __name__ == "__main__":
    main()