# Research Projects Codebase

This is a machine learning and deep learning research codebase built on JAX/Flax and Bazel. The repository contains multiple independent but infrastructure-sharing research projects, currently focusing on two main modules: **Reinforcement Learning (RL)** and **Generative Models (Generative)**.

## 📁 Codebase Structure

- **`src/projects/`**: Contains the source code for all specific research projects.
  - **`rl/`**: Reinforcement Learning module (contains implementations of DQN and related components).
  - **`generative/`**: Generative Models module (contains U-Net based generative models like DDPM diffusion models and Flow Matching).
- **`src/core/`**: Core components and infrastructure (e.g., base model classes in `model.py`, distributed training wrappers in `distributed.py`, and training state management in `train_state.py`).
- **`src/utilities/`**: General utility libraries (including logging, visualization, and training helper functions).
- **`src/data/`**: Data processing modules (e.g., data pipelines using HuggingFace `datasets`).
- **`MODULE.bazel`**: Bazel dependency and environment configuration file, managing Python versions and related package dependencies.

---

## 🛠️ Environment Setup

Before running any code, ensure that system-level dependencies are correctly loaded and the appropriate driver versions are enabled:

1. **Load CUDA and cuDNN modules** (typically required on clusters/servers):
   ```bash
   module load cuda/12.6
   module load cudnn
   ```
2. **Bazel Build System**:
   This project uses [Bazelisk](https://github.com/bazelbuild/bazelisk) (as a wrapper for Bazel) for unified build version management.
   - All Python dependencies are declared via `rules_python` in `MODULE.bazel` and are automatically fetched during the build.
   - **No need** to manually run `pip install`; Bazel will isolate the execution environment properly.

---

## 🚀 How to Run

The core way to execute code is through the `bazelisk run` command. Below is a breakdown of the "long command" you frequently use for better understanding:

### Command Breakdown

```bash
CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_LEVEL=NVL \
NCCL_SHM_DISABLE=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=.9 \
bazelisk run --config=cuda //src/projects/rl:main -- --work_dir logs/
```

- **GPU and JAX Environment Variables**:
  - `CUDA_VISIBLE_DEVICES=0`: Restricts the process to only use the GPU with index `0`.
  - `NCCL_P2P_LEVEL=NVL` & `NCCL_SHM_DISABLE=0`: Optimizes NCCL P2P shared memory and NVLink strategies for multi-GPU or single-GPU communication (essential communication optimizations for JAX).
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=.9`: Instructs JAX/XLA to pre-allocate 90% of the GPU memory to prevent OOM (Out Of Memory) or memory fragmentation.
- **Bazelisk Command**:
  - `bazelisk run --config=cuda`: Compiles and runs using the Bazel configuration with CUDA support (usually defined in the `.bazelrc` at the root directory).
  - `//src/projects/rl:main`: Specifies the Bazel Target to run. `//` represents the root directory, corresponding to the `ml_py_binary` with `name="main"` in the `src/projects/rl/BUILD` file.
- **User Arguments** (Passed to the Python script after `--`):
  - `--work_dir logs/`: Specific arguments passed to `main.py` in Python, such as paths for saving logs and models.

### Common Run Examples

**Run Reinforcement Learning (RL - DQN)**
```bash
module load cuda/12.6
module load cudnn
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL NCCL_SHM_DISABLE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.9 \
bazelisk run --config=cuda //src/projects/rl:main -- \
    --work_dir logs/rl_run \
    --num_episodes 5000 \
    --batch_size 512
```

**Run Generative Models (Generative - DDPM / Mean Flow)**
```bash
module load cuda/12.6
module load cudnn
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL NCCL_SHM_DISABLE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.9 \
bazelisk run --config=cuda //src/projects/generative:main -- \
    --work_dir logs/generative_run \
    --distributed False
```

*(Note: If you have specific `fiddle` configuration files when running `generative`, you might also need to pass them via arguments like `--experiment=xxx`, see `main.py` for details)*

---

## 🔧 Adding New Features & BUILD Files

Because this project uses **Bazel** to manage all file dependencies, **whenever you add a new `.py` file, you MUST register it in the `BUILD` file within the same directory; otherwise, Bazel won't find your new code during execution.**

### 1. Understanding `BUILD` File Structure

Open `src/projects/rl/BUILD`, and you will see two main types of Bazel Rule Macros:
- `ml_py_library`: Used to define a library file or module (to be imported by other code).
- `ml_py_binary`: Used to define an executable entry script (like `main.py`, which can be run directly via `bazelisk run`).

### 2. Steps to Add a New Feature

Suppose you create a new algorithm named `ppo.py` under `src/projects/rl/`:

1. **Register Python Library**:
   Add a `ml_py_library` block in `src/projects/rl/BUILD`:
   ```python
   ml_py_library(
       name = "ppo",
       srcs = ["ppo.py"],
       deps = [
           "flax",
           "jax",
           "optax",
           "//src/core:model",     # Depend on other local Bazel modules
       ],
   )
   ```
   > *Third-party libraries (like `flax`, `jax`) in `deps` are pre-defined in `MODULE.bazel`, so you can just write their names. Local libraries require the full path (like `//src/core:model`).*

2. **Import into the Main Program**:
   If you want to use your new algorithm in `main.py`, you must add `:ppo` to the dependencies list of `main` in the `BUILD` file.
   ```python
   ml_py_binary(
       name = "main",
       srcs = ["main.py"],
       deps = [
           # ... other existing dependencies
           ":ppo",                 # <-- Introduce the newly defined library as a dependency
       ],
   )
   ```

3. **Run Again**:
   After modifying the `BUILD` file and importing it in your code, just use the previous `bazelisk run` command. Bazel will automatically detect file changes and rebuild your execution environment.