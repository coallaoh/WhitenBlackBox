__author__ = 'joon'

import torch.nn as nn
import torch.nn.functional as F


class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(convblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class linearblock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout='none'):
        super(linearblock, self).__init__()
        self.conv = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(inplace=True)
        self.dropoutoption = dropout

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropoutoption == 'normal':
            x = self.dropout(x)
        return x


class mlpblock(nn.Module):
    def __init__(self, inputdim, outputdim, nlayer=2, hiddendim=10, activation='ReLU'):
        super(mlpblock, self).__init__()
        self.nlayer = nlayer
        self.hiddendim = hiddendim
        self.inputdim = inputdim
        self.outputdim = outputdim

        if activation == 'ReLU':
            self.act = F.relu
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(self.inputdim, self.hiddendim)
        fc_iter = []
        for n in range(self.nlayer - 2):
            fc_iter.append(linearblock(self.hiddendim, self.hiddendim))
        self.fc_iter = nn.Sequential(*fc_iter)
        self.fc_final = nn.Linear(self.hiddendim, self.outputdim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc_iter(x)
        x = self.fc_final(x)
        return x
