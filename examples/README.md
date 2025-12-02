# Examples Directory

This directory contains example scripts demonstrating empo functionality.

## Running Examples

Most examples can be run with:
```bash
PYTHONPATH=src:vendor/multigrid python examples/<script_name>.py
```

## Development Guidelines

### Quick Test Mode

**All long-running example scripts should include a `--quick` or `--fast` flag** that reduces training episodes, environments, and other time-consuming parameters. This allows developers to quickly verify scripts work without waiting for full runs.

Example:
```bash
# Full run (may take several minutes)
python examples/random_multigrid_ensemble_demo.py

# Quick test (completes in ~10 seconds)
python examples/random_multigrid_ensemble_demo.py --quick
```

When creating new examples, follow the pattern in `random_multigrid_ensemble_demo.py`:
- Use `argparse` to add `--quick/-q` flag
- Define separate constants for full vs quick mode (e.g., `NUM_EPISODES_FULL` vs `NUM_EPISODES_QUICK`)
- Document both modes in the script's docstring

## Available Examples

- `random_multigrid_ensemble_demo.py` - Demonstrates neural policy prior learning on randomly generated multigrids with 3 humans and 1 robot
