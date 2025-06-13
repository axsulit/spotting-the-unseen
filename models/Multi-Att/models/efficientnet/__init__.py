"""EfficientNet model package.

This package provides the EfficientNet model implementation and related utilities.
The EfficientNet model is a family of convolutional neural networks that achieve
state-of-the-art accuracy while being more efficient than previous models.
"""

__version__ = "0.6.3"
from .model import EfficientNet, VALID_MODELS
from .utils import (
    round_filters,
    round_repeats,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    BlockDecoder,
    GlobalParams,
    BlockArgs,
)

__all__ = [
    'EfficientNet',
    'VALID_MODELS',
    'round_filters',
    'round_repeats',
    'get_same_padding_conv2d',
    'get_model_params',
    'efficientnet_params',
    'load_pretrained_weights',
    'Swish',
    'MemoryEfficientSwish',
    'BlockDecoder',
    'GlobalParams',
    'BlockArgs',
]

