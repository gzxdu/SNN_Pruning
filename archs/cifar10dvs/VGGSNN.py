from typing import Any
import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.layer import SeqToANNContainer


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)





class VGGSNN(nn.Module):
    def __init__(self, spiking_neuron: callable = None, **kwargs: Any):
        """
        VGG 风格的 SNN 网络，采用 LIF 神经元。

        :param spiking_neuron: 选择的脉冲神经元模型（默认 `LIFNode`）
        :param kwargs: 传递给神经元的参数
        """
        super().__init__()

        if spiking_neuron is None:
            spiking_neuron = neuron.LIFNode  # 默认使用 LIF 神经元

        def conv_block(in_channels, out_channels):
            """构造卷积 + BN + Spiking 神经元的块"""
            return nn.Sequential(
                SeqToANNContainer(
                    conv3x3(in_channels, out_channels, bias=False),
                    nn.BatchNorm2d(out_channels)
                ),
                spiking_neuron(**deepcopy(kwargs))
            )

        self.feature_extractor = nn.Sequential(
            conv_block(2, 64),
            conv_block(64, 128),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),

            conv_block(128, 256),
            conv_block(256, 256),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),

            conv_block(256, 512),
            conv_block(512, 512),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),

            conv_block(512, 512),
            conv_block(512, 512),
            SeqToANNContainer(nn.AvgPool2d(2, 2))
        )

        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Flatten()),
            SeqToANNContainer(nn.Dropout(0.25)),
            SeqToANNContainer(nn.Linear(512 * 3 * 3, 100, bias=False)),  # 假设输入尺寸合适，需根据实际数据调整
            spiking_neuron(**deepcopy(kwargs)),
            SeqToANNContainer(nn.AvgPool1d(10, 10))  # Voting Layer
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor):
        """
        前向传播

        :param x: 输入张量，形状为 [N, T, C, H, W]
        :return: 输出张量，形状为 [T, N, C]
        """
        x = x.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 测试网络结构
    net = VGGSNN(spiking_neuron=neuron.LIFNode)
    print(net)
    
    x = torch.randn(16, 10, 2, 48, 48)  # [N, T, C, H, W]
    y = net(x)
    print(y.shape)  # 期望输出形状: [T, N, C]