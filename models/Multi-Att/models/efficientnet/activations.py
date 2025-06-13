"""Activation functions for EfficientNet model.

This module provides activation functions used in the EfficientNet model,
including the Swish activation function and its memory-efficient variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishImplementation(torch.autograd.Function):
    """Implementation of the Swish activation function.
    
    This class implements the forward and backward passes of the Swish
    activation function using PyTorch's autograd functionality.
    """

    @staticmethod
    def forward(ctx, i):
        """Forward pass of the Swish activation function.
        
        Args:
            ctx: Context object for storing tensors needed in backward pass.
            i (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying Swish activation.
        """
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the Swish activation function.
        
        Args:
            ctx: Context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the output.
            
        Returns:
            torch.Tensor: Gradient of the input.
        """
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    """Memory efficient implementation of the Swish activation function.
    
    This class provides a memory-efficient implementation of the Swish
    activation function by using a custom autograd function.
    """

    def forward(self, x):
        """Forward pass of the memory-efficient Swish activation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying Swish activation.
        """
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    """Standard implementation of the Swish activation function.
    
    This class provides a standard implementation of the Swish activation
    function using PyTorch's built-in operations.
    """

    def forward(self, x):
        """Forward pass of the Swish activation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying Swish activation.
        """
        return x * torch.sigmoid(x) 