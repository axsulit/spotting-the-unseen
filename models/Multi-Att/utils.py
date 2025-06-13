"""Utility functions and classes for the multiple attention model.

This module provides utility functions for distributed training, accuracy calculation,
and gradient manipulation.
"""

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist 

import torch
import torch.distributed as dist

class dist_average:
    """Class for computing distributed averages across multiple GPUs.
    
    This class handles the computation of averages across multiple GPUs in distributed
    training scenarios, with fallback to single-GPU operation when distributed training
    is not being used.
    
    Args:
        local_rank (int): The local rank of the current GPU.
    """
    
    def __init__(self, local_rank):
        """Initialize the distributed average calculator.
        
        Args:
            local_rank (int): The local rank of the current GPU.
        """
        self.rank = local_rank
        if dist.is_available() and dist.is_initialized():  # Check if distributed training is initialized
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1  # Default to 1 for single-GPU or CPU training
        self.acc = torch.zeros(1).to(local_rank)
        self.count = 0

    def step(self, input_):
        """Add a new value to the running average.
        
        Args:
            input_ (torch.Tensor or float): The value to add to the average.
        """
        self.count += 1
        if type(input_) != torch.Tensor:
            input_ = torch.tensor(input_).to(self.rank, dtype=torch.float)
        else:
            input_ = input_.detach()
        self.acc += input_

    def get(self):
        """Get the current average value.
        
        Returns:
            float: The average value across all GPUs if in distributed mode,
                  or the local average if in single-GPU mode.
        """
        # Skip distributed logic if distributed training is not used
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(self.acc, op=dist.ReduceOp.SUM)
            self.acc /= self.world_size
        return self.acc.item() / self.count


def ACC(x, y):
    """Calculate classification accuracy.
    
    Args:
        x (torch.Tensor): Model predictions (logits).
        y (torch.Tensor): Ground truth labels.
        
    Returns:
        float: Classification accuracy.
    """
    with torch.no_grad():
        a = torch.max(x, dim=1)[1]
        acc = torch.sum(a == y).float() / x.shape[0]
    #print(y,a,acc)        
    return acc

def cont_grad(x, rate=1):
    """Apply continuous gradient scaling.
    
    Args:
        x (torch.Tensor): Input tensor.
        rate (float, optional): Scaling rate for the gradient. Defaults to 1.
        
    Returns:
        torch.Tensor: Scaled tensor with modified gradient.
    """
    return rate * x + (1 - rate) * x.detach()