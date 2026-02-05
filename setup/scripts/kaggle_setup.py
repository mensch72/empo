#!/usr/bin/env python3
"""
EMPO Kaggle Setup Script

This script sets up the EMPO framework for use in Kaggle notebooks.
It handles cloning, path configuration, and dependency installation.

Usage in Kaggle:
    # First cell - clone and setup
    !git clone --depth 1 https://github.com/mensch72/empo.git
    %cd empo
    %run scripts/kaggle_setup.py
    
Or import as module:
    exec(open('scripts/kaggle_setup.py').read())
"""

import sys
import os
import subprocess


def setup_empo_paths(repo_root: str = None) -> str:
    """
    Configure Python paths for EMPO imports.
    
    Args:
        repo_root: Path to EMPO repository root. If None, uses current directory.
        
    Returns:
        The repository root path used.
    """
    if repo_root is None:
        repo_root = os.getcwd()
    
    # Add src and vendor directories to Python path
    src_path = os.path.join(repo_root, 'src')
    vendor_multigrid = os.path.join(repo_root, 'vendor', 'multigrid')
    vendor_transport = os.path.join(repo_root, 'vendor', 'ai_transport')
    
    for path in [src_path, vendor_multigrid, vendor_transport]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return repo_root


def install_dependencies(requirements_file: str = 'setup/requirements/kaggle.txt', quiet: bool = True) -> bool:
    """
    Install EMPO dependencies.
    
    Args:
        requirements_file: Path to requirements file.
        quiet: If True, suppress pip output.
        
    Returns:
        True if installation succeeded.
    """
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements_file]
    if quiet:
        cmd.insert(4, '-q')
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL if quiet else None)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: pip install failed: {e}")
        return False


def detect_environment() -> dict:
    """
    Detect the current notebook environment.
    
    Returns:
        Dictionary with environment info.
    """
    import torch
    
    env_info = {
        'platform': 'unknown',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_memory_gb': None,
        'pytorch_version': torch.__version__,
        'python_version': sys.version.split()[0],
    }
    
    # Detect platform
    if os.path.exists('/kaggle'):
        env_info['platform'] = 'kaggle'
    elif 'google.colab' in sys.modules or os.path.exists('/content'):
        env_info['platform'] = 'colab'
    elif os.path.exists('/home/jovyan'):
        env_info['platform'] = 'jupyter'
    else:
        env_info['platform'] = 'local'
    
    # GPU info
    if env_info['gpu_available']:
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return env_info


def verify_imports() -> dict:
    """
    Verify that all EMPO imports work.
    
    Returns:
        Dictionary mapping module names to success status.
    """
    results = {}
    
    # Core EMPO
    try:
        results['empo_core'] = True
    except ImportError as e:
        results['empo_core'] = str(e)
    except Exception as e:
        results['empo_core'] = f"Error: {type(e).__name__}: {e}"
    
    # MultiGrid
    try:
        results['multigrid'] = True
    except ImportError as e:
        results['multigrid'] = str(e)
    except Exception as e:
        results['multigrid'] = f"Error: {type(e).__name__}: {e}"
    
    # Environments
    try:
        results['envs'] = True
    except ImportError as e:
        results['envs'] = str(e)
    except Exception as e:
        results['envs'] = f"Error: {type(e).__name__}: {e}"
    
    # Phase 2 (neural networks) - may fail due to tensorboard issues
    try:
        results['phase2'] = True
    except ImportError as e:
        results['phase2'] = str(e)
    except Exception as e:
        # TensorBoard/protobuf issues are common on Kaggle
        results['phase2'] = f"Warning: {type(e).__name__} (TensorBoard issue, try restarting kernel)"
    
    return results


def print_status(repo_root: str, env_info: dict, import_results: dict):
    """Print setup status summary."""
    print("=" * 60)
    print("EMPO Setup Complete")
    print("=" * 60)
    print(f"\nRepository: {repo_root}")
    print(f"Platform: {env_info['platform']}")
    print(f"Python: {env_info['python_version']}")
    print(f"PyTorch: {env_info['pytorch_version']}")
    
    if env_info['gpu_available']:
        print(f"GPU: {env_info['gpu_name']} ({env_info['gpu_memory_gb']:.1f} GB)")
    else:
        print("GPU: Not available (CPU mode)")
    
    print("\nImport Status:")
    all_ok = True
    for module, status in import_results.items():
        if status is True:
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}: {status}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All imports successful! Ready to use EMPO.")
    else:
        print("\n⚠ Some imports failed. Check error messages above.")
    
    print("=" * 60)


def fix_protobuf_issues():
    """
    Fix protobuf/tensorboard compatibility issues common on Kaggle/Colab.
    Should be called before importing tensorboard-dependent modules.
    """
    try:
        # Force protobuf to use pure-python implementation to avoid C++ issues
        os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    except Exception as exc:
        # Best-effort: failing to tweak this env var should not break setup,
        # but we log a warning so issues are not silently hidden.
        print(
            f"Warning: could not set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: {exc}",
            file=sys.stderr,
        )


def setup(install_deps: bool = True, quiet: bool = True) -> dict:
    """
    Complete EMPO setup for Kaggle/Colab notebooks.
    
    Args:
        install_deps: If True, install dependencies from requirements file.
        quiet: If True, suppress installation output.
        
    Returns:
        Dictionary with setup results.
    """
    repo_root = setup_empo_paths()
    
    # Fix protobuf issues before any imports
    fix_protobuf_issues()
    
    # Choose requirements file based on platform
    if os.path.exists('/kaggle'):
        req_file = 'setup/requirements/kaggle.txt'
    else:
        req_file = 'setup/requirements/colab.txt'
    
    if install_deps and os.path.exists(req_file):
        print(f"Installing dependencies from {req_file}...")
        install_dependencies(req_file, quiet=quiet)
        print("✓ Dependencies installed")
        print("  Note: If you see TensorBoard errors, restart the kernel and re-run.")
    
    env_info = detect_environment()
    import_results = verify_imports()
    
    print_status(repo_root, env_info, import_results)
    
    return {
        'repo_root': repo_root,
        'env_info': env_info,
        'imports': import_results,
    }


# Auto-run when executed directly or via %run
if __name__ == '__main__' or '__file__' not in dir():
    setup()
