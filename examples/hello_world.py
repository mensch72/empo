"""
Minimal example that imports PyTorch and prints a greeting.
Useful for testing Kaggle/Colab environment setup.
"""

import torch

print("Hello World!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
