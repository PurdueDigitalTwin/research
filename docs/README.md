# Research Monorepo

A monolithic source code repository designed for multi-language, multi-platform, reproducible, and scalable development. This repository manages research projects, core libraries, and data pipelines using **Bazel** for build hermeticity and **Python** for algorithm development.

## 🚀 Getting Started

### Prerequisites

- [Bazel](https://bazel.build/install) (via Bazelisk recommended)
- Python 3.x
- A compatible accelerator driver (CUDA for NVIDIA, or standard drivers for MPS/TPU)

## Project Structure

The repository structure is shown here:

```text
.
├── CODEOWNERS             # definition of code ownership and review gates
├── MODULE.bazel           # Bazel module definitions and dependency locking
├── docs/                  # Documentation and maintenance guides
├── src/
│   ├── core/              # Shared, fundamental algorithms and base classes
│   ├── data/              # Data loading, processing pipelines, and dataset managers
│   ├── projects/          # Individual research experiments (e.g., RL agents, paper implementations)
│   ├── utilities/         # distinct helper tools, loggers, and visualization scripts
│   └── py.typed           # Marker file for PEP 561 type checking compliance
└── third_party/           # Vendor code and external package management
    ├── requirements.in    # Base Python dependencies
    ├── requirements_*.in  # Hardware-specific dependencies (CUDA, MPS, TPU)
    └── defs.bzl           # Custom Bazel definitions for third-party tools
```
