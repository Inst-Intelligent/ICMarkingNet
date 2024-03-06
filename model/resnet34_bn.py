import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights
from collections import namedtuple


class resnet34_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(resnet34_bn, self).__init__()

        parameters = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None

        pretrained_features = torchvision.models.resnet34(weights = parameters)

        self.slice0 = torch.nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True)
            pretrained_features.conv1,
            pretrained_features.bn1,
            pretrained_features.relu,
        )

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        self.slice1.add_module("layer1", pretrained_features.layer1)
        self.slice2.add_module("layer2", pretrained_features.layer2)
        self.slice3.add_module("layer3", pretrained_features.layer3)
        self.slice4.add_module("layer4", pretrained_features.layer4)

        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        if freeze:
            for param in self.slice0.parameters():  # only first conv
                param.requires_grad = False
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        # X: H, W, 3

        h = self.slice0(X)  # h: H/2, W/2, 64
        h = self.slice1(h)  # h: H/4, W/4, 64
        out1 = h
        h = self.slice2(h)  # h: H/8, W/8, 128
        out2 = h
        h = self.slice3(h)  # h: H/16, W/16, 256
        out3 = h
        h = self.slice4(h)  # h: H/32, W/32, 512
        out4 = h
        h = self.slice5(h)  # h: H/64, W/64, 1024
        out5 = h

        resnet_outputs = namedtuple(
            "ResNetOutputs", ["out5", "out4", "out3", "out2", "out1"]
        )
        out = resnet_outputs(out5, out4, out3, out2, out1)
        return out

        out = x
