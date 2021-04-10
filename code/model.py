import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

from .activation import Identity
from .activation import Activation


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        # one kernel make one feature map
        # kernel 厚度 = channel 數
        self.is_changed = in_channels != out_channels
        self.trans = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=stride)

    def forward(self, x):
        f_x = self.conv1(x)
        f_x = self.bn1(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv2(f_x)
        f_x = self.bn2(f_x)

        if self.is_changed:
            x = self.trans(x)

        x = f_x + x
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=4):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels, out_channels=dilation*out_channels, kernel_size=1, padding=0)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * dilation)

        self.relu = nn.ReLU(inplace=True)

        # one kernel make one feature map
        # kernel 厚度 = channel 數
        self.is_changed = in_channels != (out_channels*dilation)
        self.trans = nn.Conv2d(in_channels, out_channels *
                               dilation, kernel_size=1, stride=stride)

    def forward(self, x):

        f_x = self.conv1(x)
        f_x = self.bn1(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv2(f_x)
        f_x = self.bn2(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv3(f_x)
        f_x = self.bn3(f_x)

        if self.is_changed:
            x = self.trans(x)

        x = f_x + x
        x = self.relu(x)
        return x


class _ResNet(nn.Module):
    def __init__(self, block, block_cnts, dilation=1):
        super(_ResNet, self).__init__()

        self.in_channels = 64
        self.out_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._layer(
            block, block_cnts[0], dilation, self.in_channels, self.out_channels, stride=1)
        self.layer2 = self._layer(
            block, block_cnts[1], dilation, self.in_channels, self.out_channels, stride=2)
        self.layer3 = self._layer(
            block, block_cnts[2], dilation, self.in_channels, self.out_channels, stride=2)
        self.layer4 = self._layer(
            block, block_cnts[3], dilation, self.in_channels, self.out_channels, stride=2)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.dense = nn.Linear(in_features=self.in_channels, out_features=10)
        self.softmax = nn.Softmax(dim=1)
        self.flatern = nn.Flatten(start_dim=1)

    def _layer(self, block, block_cnt, dilation, in_channels, out_channels, stride):
        # in_channels: param of previous block output channel
        # out_channels: param of current block input channel

        blocks = []
        blocks.append(
            block(in_channels=in_channels,
                  out_channels=out_channels, stride=stride)
        )

        for cnt in range(1, block_cnt):
            b = block(in_channels=dilation * out_channels,
                      out_channels=out_channels)
            blocks.append(b)

        self.in_channels = out_channels * dilation
        self.out_channels = out_channels * 2

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg(x)

        x = self.flatern(x)
        x = self.dense(x)

        x = self.softmax(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(
            1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(
            p=dropout, inplace=True) if dropout else Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Extractor(nn.Module):
    def __init__(self, in_channels, out_channels,  use_batchnorm=True, maxpool=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            Conv2dReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                use_batchnorm=use_batchnorm,
            ),
            nn.AvgPool2d((1, 2)) if maxpool else Identity(),
            Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                use_batchnorm=use_batchnorm,
            )
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    def __init__(self, classes, input_size=4, hidden_size=64, num_layers=32, batch_first=True):
        super(Decoder, self).__init__()

        # self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #                    num_layers=num_layers, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=4, nhead=2, num_encoder_layers=6)
        self.linear1 = nn.Linear(
            in_features=400, out_features=hidden_size//2, bias=True)
        self.linear2 = nn.Linear(
            in_features=hidden_size//2, out_features=classes, bias=True)
        self.flatten = nn.Flatten(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = torch.squeeze(x, 1)    # batch, 100, 4

        # uncomment here to use transformer
        #######################################
        # transformer batch is second
        # x = torch.movedim(x, 0, 1)

        # x = self.transformer.encoder(x)

        # # transformer batch is second
        # x = torch.movedim(x, 0, 1)
        #######################################

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):

        latent = self.encoder(x)
        latent = torch.transpose(latent, 2, 3)
        latent = torch.squeeze(latent, 1)
        return latent, self.decoder(latent)
