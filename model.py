'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import roi
from resnet34_bn import resnet34_bn

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Spotting(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(Spotting, self).__init__()

        self.basenet = resnet34_bn(pretrained, freeze)
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(256, 256, 128)
        self.upconv3 = double_conv(128, 128, 64)
        self.upconv4 = double_conv(64, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):

        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(
            y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(
            y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[3]], dim=1)
        feature_d4 = y
        y = self.upconv3(y)
        y = F.interpolate(
            y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature_d4
    
class Direction(nn.Module):
    def __init__(self, num_classes=2):

        super(Direction, self).__init__()

        def _conv_block(in_channels, out_channels):
            return nn.Sequential(OrderedDict(
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                bn = nn.BatchNorm2d(out_channels),
                relu = nn.ReLU(inplace=True),
                pool = nn.MaxPool2d(2, 2)
            ))

        # feature extraction
        self.add_module('cbr1', _conv_block(256, 256))
        self.add_module('cbr2', _conv_block(256, 128))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(2)

        # decoupled angle representation
        self.angle = nn.Linear(128 * 4, num_classes)
        self.sign = nn.Linear(128 * 4, num_classes)
    
    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) 
        
        ang = self.angle(x)
        sgn = self.sign(x)

        return ang, sgn


class MarkingHead(nn.Module):
    def __init__(
        self,
        map_height=8,
        num_class=len(roi.CHARS) + 1,
        map_to_seq_hidden=64,
        rnn_hidden=256,
    ):
        super(MarkingHead, self).__init__()

        # Convolutional layers
        def _cnn():
            channels = [256, 256, 256, 512, 1024]
            kernel_sizes = [3, 3, 3, 2]
            strides = [1, 1, 1, 1]
            paddings = [1, 1, 1, 0]

            cnn = nn.Sequential()

            def conv_relu(i, batch_norm=False):
                input_channel = channels[i]
                output_channel = channels[i + 1]
                cnn.add_module(
                    f"conv{i}",
                    nn.Conv2d(
                        input_channel,
                        output_channel,
                        kernel_sizes[i],
                        strides[i],
                        paddings[i],
                    ),
                )

                if batch_norm:
                    cnn.add_module(f"bn{i}", nn.BatchNorm2d(output_channel))
                cnn.add_module(f"relu{i}", nn.ReLU(inplace=True))

            conv_relu(0)
            cnn.add_module("pooling2", nn.MaxPool2d(kernel_size=(2, 1))) 

            conv_relu(1, batch_norm=True)
            conv_relu(2, batch_norm=True)
            cnn.add_module("pooling3", nn.MaxPool2d(kernel_size=(2, 1))) 
            conv_relu(3) 
            return cnn, channels[-1]
    
        self.cnn, output_channel, = _cnn()
        output_height = map_height // 8
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        # Recurrent layers
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        # Predction
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, images):
        x = self.cnn(images)
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1) 
        x = self.map_to_seq(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)

        return self.dense(x)
    
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.spotting = Spotting()
        self.direction = Direction()
        self.head = MarkingHead(map_height=8)
        self.device = torch.device('cpu')
        
    def setDevice(self, device):
        self.device = device
        self.to(self.device)

    def forward(self, inputs):

        images, saliency_labels, more_info = inputs
        self.spotting.eval()
        saliency_maps, shared_features = self.spotting(images.to(self.device))
        
        words_roi, a_hat, p_hat, m_hat, m_len = roi.find_word_boxes_with_labels(
            saliency_maps[:, :, :, 0], 
            saliency_maps[:, :, :, 1], 
            more_info
        )

        links = roi.find_links(saliency_maps[:, :, :, 1])
        squared_word_featuers = roi.roi_pooling(shared_features.detach(), words_roi)
        axiality, postivity = self.direction(squared_word_featuers)

        word_features = roi.link_sampling(shared_features, words_roi, links, axiality, postivity)
        
        logits = self.head(word_features)
        marking_probs = torch.nn.functional.log_softmax(logits, dim=2)

        return (
            (
                (saliency_maps[:, :, :, 0], saliency_maps[:, :, :, 1]),
                (axiality, postivity), marking_probs),
            (saliency_labels,(a_hat, p_hat), (m_hat, m_len)), words_roi
        )
