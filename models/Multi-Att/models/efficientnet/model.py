"""EfficientNet model implementation.

This module implements the EfficientNet architecture, which is a family of convolutional
neural networks that achieve state-of-the-art accuracy while being more efficient than
previous models.
"""

import torch
from torch import nn
from torch.nn import functional as F
import kornia
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    
    This block implements the MBConv block used in EfficientNet, which consists of
    an expansion phase, depthwise convolution, and projection phase.
    
    Args:
        block_args (namedtuple): BlockArgs containing block configuration.
        global_params (namedtuple): GlobalParams containing model configuration.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """Forward pass of the MBConv block.
        
        Args:
            inputs (torch.Tensor): Input tensor.
            drop_connect_rate (float, optional): Drop connect rate. Defaults to None.
            
        Returns:
            torch.Tensor: Output tensor after block processing.
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        
        Args:
            memory_efficient (bool, optional): Whether to use memory efficient swish.
                Defaults to True.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
    
    This class implements the EfficientNet architecture, which is a family of
    convolutional neural networks that achieve state-of-the-art accuracy while
    being more efficient than previous models.
    
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
        escape (str, optional): Layer to escape to during forward pass. Defaults to ''.
    """

    def __init__(self, blocks_args=None, global_params=None, escape=''):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.escape = escape
        self._global_params = global_params
        self._blocks_args = blocks_args
        
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        self.stage_map = []
        stage_count = 0
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            stage_count += 1
            self.stage_map += [''] * (block_args.num_repeat - 1)
            self.stage_map.append('b%s' % stage_count)
            
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        
        Args:
            memory_efficient (bool, optional): Whether to use memory efficient swish.
                Defaults to True.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs, layers):
        """Extract features from the model.
        
        Args:
            inputs (torch.Tensor): Input tensor.
            layers (dict): Dictionary to store intermediate layer outputs.
            
        Returns:
            torch.Tensor or None: Final feature map or None if escape layer is reached.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        layers['b0'] = x
        
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            stage = self.stage_map[idx]
            if stage:     
                layers[stage] = x
                if stage == self.escape:
                    return None
                    
        # Head
        x = self._bn1(self._conv_head(x))
        x = self._swish(x)
        return x

    def forward(self, x):
        """Forward pass of the EfficientNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            dict: Dictionary containing intermediate layer outputs and final logits.
        """
        bs = x.size(0)
        layers = {}
        x = self.extract_features(x, layers)
        if x is None:
            return layers
        layers['final'] = x
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        layers['logits'] = x
        return layers

    @classmethod
    def from_name(cls, model_name, override_params=None, escape=''):
        """Create an EfficientNet model from a model name.
        
        Args:
            model_name (str): Name of the EfficientNet model.
            override_params (dict, optional): Parameters to override. Defaults to None.
            escape (str, optional): Layer to escape to. Defaults to ''.
            
        Returns:
            EfficientNet: A new EfficientNet model instance.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, escape)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3, escape=''):
        """Create an EfficientNet model from pretrained weights.
        
        Args:
            model_name (str): Name of the EfficientNet model.
            advprop (bool, optional): Whether to use adversarial training. Defaults to False.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            escape (str, optional): Layer to escape to. Defaults to ''.
            
        Returns:
            EfficientNet: A new EfficientNet model instance with pretrained weights.
        """
        model = cls.from_name(model_name, override_params={'num_classes': num_classes}, escape=escape)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given model name.
        
        Args:
            model_name (str): Name of the EfficientNet model.
            
        Returns:
            int: Input image size.
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.
        
        Args:
            model_name (str): Name of the EfficientNet model.
            
        Raises:
            ValueError: If model name is not valid.
        """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
