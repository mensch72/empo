# EMPO Notebooks

This directory contains Jupyter notebooks for running EMPO on cloud platforms.

## Quick Start: Launcher Notebooks

For running example scripts with minimal setup, use the launcher notebooks:

### kaggle_launcher.ipynb (Recommended)

Minimal 5-cell notebook for running any example script on Kaggle:
1. Clone repo
2. Setup paths
3. `%run examples/<script>.py`
4. Copy outputs
5. Download

**Best for**: Long-running training with background execution.

### colab_launcher.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mensch72/empo/blob/main/notebooks/colab_launcher.ipynb)

Minimal launcher for Google Colab - same structure as Kaggle launcher.

## Tutorial Notebooks

For learning the EMPO framework interactively:

### empo_colab_demo.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mensch72/empo/blob/main/notebooks/empo_colab_demo.ipynb)

Comprehensive tutorial covering:

- **Setup**: Installing dependencies and configuring the environment
- **Environment Exploration**: Creating and visualizing MultiGrid environments
- **State Management**: Using `get_state()` and `set_state()` for exact state control
- **DAG Computation**: Computing the complete state-space structure
- **Policy Priors**: Computing human policy priors via backward induction
- **Visualization**: Rendering environment states and episode animations

## Running Notebooks Locally

You can also run these notebooks locally using Jupyter:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter server
jupyter notebook

# Or with JupyterLab
pip install jupyterlab
jupyter lab
```

## Creating New Notebooks

When creating new notebooks for this repository:

1. **Docker container (recommended)**: When running notebooks inside the Docker container (`make shell`, then `jupyter notebook`), the `PYTHONPATH` is already set and no path setup is needed.

2. **Google Colab**: When running in Colab, set up Python paths at the start:
   ```python
   import sys, os
   sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
   sys.path.insert(0, os.path.join(os.getcwd(), 'vendor', 'multigrid'))
   ```

3. **Import from empo package**:
   ```python
   from empo import WorldModel, PossibleGoal, compute_human_policy_prior
   from envs.one_or_three_chambers import SmallOneOrThreeChambersMapEnv
   ```

3. **Add Colab badge** if the notebook should work in Colab:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mensch72/empo/blob/main/notebooks/YOUR_NOTEBOOK.ipynb)
   ```

## Platform Limitations

### Common to Both Platforms

- **No `--async` mode**: Multiprocessing doesn't work in notebooks
- **No MPI**: Use `parallel=False` for backward induction

### Google Colab

- **Session timeout**: Free tier disconnects after ~90 min idle
- **No background execution**: Session stops when browser closes (free tier)
- **Dynamic quota**: Heavy users may be throttled

### Kaggle

- **Internet required**: Must enable in Settings → Internet → On
- **GPU quota**: 30 hours/week (T4 or P100)
- **Session limit**: Max 12 hours per session
- **Background execution**: ✅ "Save & Run All" continues after browser closes

## Running Example Scripts

Use the launcher notebooks to run any script from [examples/](../examples/):

```python
# Quick demos
%run examples/simple_example.py
%run examples/state_management_demo.py

# With flags
%run examples/phase2_robot_policy_demo.py --quick
%run examples/phase2_robot_policy_demo.py --ensemble --tabular
```

See [examples/README.md](../examples/README.md) for the full list.

## Running Locally

```bash
pip install jupyter
jupyter notebook
```

See the main [README.md](../README.md) for more information.
