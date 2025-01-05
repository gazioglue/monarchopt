# MonarchOpt: Monarch Butterfly Swarm Optimization

A Python implementation of the Monarch Swarm Optimization algorithm, designed for solving binary optimization problems. The algorithm is inspired by the migration behavior of monarch butterflies and uses a novel approach combining swarm intelligence with gradient-based optimization.

## Features

- Binary optimization for various problem types
- Built-in command line interface
- Automatic result saving and history tracking
- Early stopping with known optimum
- Automatic progress reporting
- Built-in timeout mechanism
- Reproducible results with seed setting

## Installation

```bash
pip install monarchopt
```

## Quick Start

### Basic Usage

```python
from monarchopt import MSO
import numpy as np

def fitness(solution):
    """Example fitness function: maximize sum of elements."""
    return np.sum(solution)

MSO.run(
    obj_func=fitness,
    dim=20,
    pop_size=50,
    max_iter=100,
    obj_type='max'
)
```

### Solving DUF Benchmark Functions

The package includes standalone scripts for solving DUF (Decomposable Unitation-based Functions) problems:

```bash
# Basic usage
python solve_dufs.py duf1

# With custom parameters
python solve_dufs.py duf2 --dim 200 --pop-size 2000 --seed 42
```

### Solving UFLP Problems

For solving Uncapacitated Facility Location Problems:

```bash
# Basic usage
python solve_uflp.py cap71.txt

# With custom parameters
python solve_uflp.py cap71.txt --pop-size 2000 --max-iter 1000 --seed 42
```

## Documentation

For more detailed usage instructions and examples, see [USAGE.md](USAGE.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.