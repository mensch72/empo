# EMPO Notebooks

This directory contains Jupyter notebooks for interactive exploration of the EMPO framework.

## Available Notebooks

### empo_colab_demo.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mensch72/empo/blob/main/notebooks/empo_colab_demo.ipynb)

A comprehensive Google Colab demo that covers:

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

## Colab Limitations

Some features have limitations in Google Colab:

- **MPI**: Not supported - use `parallel=False` for backward induction
- **Docker**: Not available in Colab environment
- **Session timeout**: Free tier sessions timeout after ~12 hours
- **Large state spaces**: May require reducing `max_steps` for memory constraints

See the main [README.md](../README.md) for more information.
