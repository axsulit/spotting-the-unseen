"""EfficientNet model command line interface.

This module provides a command line interface for the EfficientNet model,
allowing users to create and test models directly from the command line.
"""

import torch
from .model import EfficientNet
from .utils import BlockDecoder, GlobalParams, BlockArgs, efficientnet_params


def main():
    """Main function to demonstrate EfficientNet model usage.
    
    This function creates an EfficientNet-B0 model and runs a forward pass
    with a random input tensor to demonstrate basic functionality.
    """
    # Create model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.eval()

    # Create random input
    x = torch.randn(1, 3, 224, 224)

    # Run forward pass
    with torch.no_grad():
        output = model(x)
    print(output['logits'].shape)  # Should be [1, 1000]


if __name__ == '__main__':
    main() 