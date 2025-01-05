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

```python
from monarchopt import MSO
import numpy as np

def simple_fitness(solution):
    """Example fitness function: maximize sum of elements."""
    return np.sum(solution)

MSO.run(
    obj_func=simple_fitness,
    dim=100,
    pop_size=1000,
    max_iter=800,
    obj_type='max',
    neighbour_count=3,
    gradient_strength=0.8,
    base_learning_rate=0.1
)
```

## Test Examples and Data

To run the test examples (UFLP and DUF problems):

1. Clone the GitHub repository:
```bash
git clone https://github.com/gazioglue/monarchopt.git
cd monarchopt
```

2. Run UFLP solver:
```bash
python examples/solve_uflp.py examples/data/uflp/test_instances/cap71.txt
```

3. Run DUF solver:
```bash
python examples/solve_dufs.py duf1
```

### Available Command Line Options

For UFLP:
```bash
python solve_uflp.py cap71.txt --pop-size 2000 --max-iter 1000 --seed 42
```

For DUF:
```bash
python solve_dufs.py duf2 --dim 200 --pop-size 2000 --seed 42
```

## Documentation

For more detailed usage instructions and examples, see [USAGE.md](USAGE.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.