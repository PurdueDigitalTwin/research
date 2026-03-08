# Reinforcement Learning Algorithms

Authors: [_Yaguang Li_](https://github.com/mrbyflyg), [_Juanwu Lu_](https://github.com/juanwulu)

Last Updated: 2026-03-08

## Overview

This directory contains implementations of various reinforcement learning algorithms. Each algorithm is implemented in its own subdirectory, with a README file that provides an overview of the algorithm, its implementation details, and how to run it.

## Getting Started

The folder structure is organized as follows:

```plain
src/projects/rl
├── BUILD
├── README.md
├── agents
│   ├── BUILD
│   └── dqn.py
├── common.py
├── environment
├── main.py
└── replay_buffer.py
```

- The `agents` directory contains implementations of different RL policies. If you would like to add a new algorithm, follow the instructions in [New Algorithm Implementation](#new-algorithm-implementation) below.
- The `environment` directory contains the wrapped environments that inherit from the `BaseEnv` class defined in `common.py`. To add a new environment, follow the instructions in [New Environment Implementation](#new-environment-implementation) below.
- The `main.py` file serves as the entry point for training and evaluating the algorithms.

### New Algorithm Implementation

Please follow the steps below to implement a new reinforcement learning algorithm:

1. Create a new Python file in the `agents` directory, for example, `my_algorithm.py`.
2. Implement the algorithm by defining a new class that inherits from the `BaseAgent` class defined in `common.py`. This class should implement the necessary methods for training and action selection.
3. Please pay careful attention to the documentation and comments in the `BaseAgent` class for guidance on implementing the algorithm, especially regarding the proper overriding of methods and the expected input/output formats.

### New Environment Implementation

TBD
