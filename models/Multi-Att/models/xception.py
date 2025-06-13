"""Xception model implementation for deepfake detection.

This module implements the Xception architecture, which is a deep convolutional neural
network architecture inspired by Inception, where Inception modules have been replaced
with depthwise separable convolutions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class SeparableConv2d(nn.Module):
    """Separable 2D convolution layer.
    
    This layer performs a depthwise separable convolution, which consists of a
    depthwise convolution followed by a pointwise convolution. This reduces the
    number of parameters and computation compared to a standard convolution.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple, optional): Size of the convolving kernel. Defaults to 1.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        """Forward pass of the separable convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor after separable convolution.
        """
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    """Xception block containing multiple separable convolutions.
    
    This block implements a series of separable convolutions with optional skip
    connections and batch normalization.
    
    Args:
        in_filters (int): Number of input filters.
        out_filters (int): Number of output filters.
        reps (int): Number of repetitions of the separable convolution.
        strides (int, optional): Stride of the first convolution. Defaults to 1.
        start_with_relu (bool, optional): Whether to start with ReLU. Defaults to True.
        grow_first (bool, optional): Whether to grow filters first. Defaults to True.
    """

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        """Forward pass of the Xception block.
        
        Args:
            inp (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after block processing.
        """
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class xception(nn.Module):
    """Xception model for deepfake detection.
    
    This class implements the Xception architecture, which is a deep convolutional
    neural network that uses depthwise separable convolutions instead of standard
    convolutions.
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 2.
        pretrained (str, optional): Whether to use pretrained weights. Can be 'imagenet'
            or path to weights file. Defaults to 'imagenet'.
        escape (str, optional): Layer to escape to during forward pass. Defaults to ''.
    """

    def __init__(self, num_classes=2, pretrained='imagenet', escape=''):
        super(xception, self).__init__()
        self.escape = escape
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Block definitions
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        
        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(2048, num_classes)

        # Define sequence of layers for forward pass
        self.seq = []
        self.seq.append(('b0', [self.conv1, lambda x: self.bn1(x), self.relu1, self.conv2, lambda x: self.bn2(x)]))
        self.seq.append(('b1', [self.relu2, self.block1]))
        self.seq.append(('b2', [self.block2]))
        self.seq.append(('b3', [self.block3]))
        self.seq.append(('b4', [self.block4]))
        self.seq.append(('b5', [self.block5]))
        self.seq.append(('b6', [self.block6]))
        self.seq.append(('b7', [self.block7]))
        self.seq.append(('b8', [self.block8]))
        self.seq.append(('b9', [self.block9]))
        self.seq.append(('b10', [self.block10]))
        self.seq.append(('b11', [self.block11]))
        self.seq.append(('b12', [self.block12]))
        self.seq.append(('final', [self.conv3, lambda x: self.bn3(x), self.relu3, self.conv4, lambda x: self.bn4(x)]))
        self.seq.append(('logits', [self.relu4, lambda x: F.adaptive_avg_pool2d(x, (1, 1)), lambda x: x.view(x.size(0), -1), self.last_linear]))

        # Load pretrained weights if specified
        if pretrained == 'imagenet':
            self.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'), strict=False)
        elif pretrained:
            ckpt = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(ckpt['state_dict'])
        else:
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        """Forward pass of the Xception model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            dict: Dictionary containing intermediate layer outputs and final logits.
        """
        layers = {}
        for stage in self.seq:
            for f in stage[1]:
                x = f(x)
            layers[stage[0]] = x
            if stage[0] == self.escape:
                break
        return layers
