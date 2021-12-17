import torch
from torch import nn
import torch.nn.functional as F


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(ResUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.batch_norm = batch_norm
        prev_channels = in_channels

        # residual concat preparation
        self.conv = nn.Conv2d(in_channels, 2 ** wf, kernel_size=3, padding=int(padding))
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(2 ** wf)
        self.relu = nn.ReLU()

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetResConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        '''
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x) # leaky relu maybe better
        '''
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetResConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetResConvBlock, self).__init__()

        if batch_norm:
            bn = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding))
        self.bn1 = bn(out_size)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding))
        self.bn2 = bn(out_size)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding))
        self.bn3 = bn(out_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        identity = x
        
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu3(out)
        
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetResConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
