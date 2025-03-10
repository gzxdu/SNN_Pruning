from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import functional, layer, surrogate, neuron

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)





# x = torch.rand([2, 2, 128, 128])
# net = DVSGestureNet(128, neuron.LIFNode, surrogate_function=surrogate.ATan())
# for name,param in net.named_parameters():
#     print(name)
#     print(param.numel())
# functional.reset_net(net)
# functional.set_step_mode(net, 'm')
# x = torch.rand([4, 2, 2, 128, 128])
# print(net(x).shape)
# functional.reset_net(net)
