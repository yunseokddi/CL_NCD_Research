from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import pickle
import os.path
import datetime
import numpy as np


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_labeled_classes=5, num_unlabeled_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head1 = nn.Linear(512 * block.expansion, num_labeled_classes)
        self.head2 = nn.Linear(512 * block.expansion, num_unlabeled_classes)

        self.l2_classifier = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_feat(self, feat):
        out = feat
        # out = F.relu(out)  # add ReLU to benifit ranking
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
        else:
            out1 = self.head1(out)
        # out2 = self.head2(out)
        return out1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if out.size(2) > 4:
            out = F.avg_pool2d(out, out.size(2))
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(out, inplace=False)  # add ReLU to benifit ranking
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
            out1 = out1.clone().detach() / 0.1
        else:
            out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2, out


class ResNetTri(nn.Module):
    def __init__(self, block, num_blocks, num_labeled_classes=80, num_unlabeled_classes1=10, num_unlabeled_classes2=10):
        super(ResNetTri, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head1 = nn.Linear(512 * block.expansion, num_labeled_classes)
        self.head2 = nn.Linear(512 * block.expansion, num_unlabeled_classes1)
        self.head3 = nn.Linear(512 * block.expansion, num_unlabeled_classes2)

        self.l2_classifier = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_feat(self, feat):
        out = feat
        # out = F.relu(out)  # add ReLU to benifit ranking
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
        else:
            out1 = self.head1(out)
        # out2 = self.head2(out)
        return out1

    def forward(self, x, output='None'):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if out.size(2) > 4:
            out = F.avg_pool2d(out, out.size(2))
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(out, inplace=False)
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
            out1 = out1.clone().detach() / 0.1
        else:
            out1 = self.head1(out)
        out2 = self.head2(out)
        out3 = self.head3(out)
        if output == 'test':
            return out1, out2, out3, out
        else:
            return out1, out3, out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion * planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out = out.clone().detach() + torch.cat([shortcut, torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)], 1)
            # out += torch.cat([shortcut,torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)],1)
        else:
            out = out.clone().detach() + self.shortcut(x)
            # out += self.shortcut(x)
        out = F.relu(out, inplace=False)
        return out