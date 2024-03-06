import torch
import torch.nn as nn
import torch.nn.functional as F

from model.char_spot import Spotting
from model.direc_recog import Direction
from model.mark_head import MarkingHead

import model.roi as roi

import time

count = 0

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.spotting = Spotting()
        self.direction = Direction()
        self.head = MarkingHead()

        self.device = torch.device('cpu')
        

    def setDevice(self, device):
        
        self.device = device
        roi.init(self.device)
        self.to(self.device)


    def forward(self, inputs):

        global count
        angles = [[180, 0], [270, 90]]

        images, saliency_labels, more_info = inputs
        
        self.spotting.eval()

        saliency_maps, shared_features = self.spotting(images.to(self.device))
        
        words_roi, a_hat, p_hat, m_hat, m_len = roi.find_word_boxes_with_labels(
            saliency_maps[:, :, :, 0], 
            saliency_maps[:, :, :, 1], 
            more_info
        )

        links = roi.find_links(saliency_maps[:, :, :, 1])

        squared_word_featuers = roi.roiPooling(shared_features.detach(), words_roi)
        axiality, postivity = self.direction(squared_word_featuers)

        word_features = roi.linkSampling(shared_features, words_roi, links, axiality, postivity)

        logits = self.head(word_features)
        marking_probs = torch.nn.functional.log_softmax(logits, dim=2)

        return (
            ((saliency_maps[:, :, :, 0], saliency_maps[:, :, :, 1]), (axiality, postivity), marking_probs),
            (saliency_labels,(a_hat, p_hat), (m_hat, m_len)), words_roi
        )
