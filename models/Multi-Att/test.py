"""Test script for checking PyTorch installation and CUDA availability.

This script verifies the PyTorch installation and CUDA availability by:
1. Printing the PyTorch version
2. Checking if CUDA is available
3. Clearing the CUDA cache
"""

import torch
print(torch.__version__)  # Check the PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is available

torch.cuda.empty_cache()
