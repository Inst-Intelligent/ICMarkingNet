'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''

import math
import torch
import torch.nn as nn

class Losses(nn.Module):

    def __init__(self, sigma = 0.5, alpha = 0.8, n_iteration = 10000, is_train = True):
        super(Losses, self).__init__()

        # for text and link maps
        self.mse = SaliencyLoss() 

        # for positivity and axiality
        self.ce1 = nn.CrossEntropyLoss() 
        self.ce2 = nn.CrossEntropyLoss()

        # for marking recognition
        self.ctc = nn.CTCLoss(reduction="mean", zero_infinity=True) 
        self.sigma = sigma
        self.alpha = alpha
        self.n = n_iteration

        self.is_train = is_train

    def setDevice(self, device):
        self.device = device
        self.to(self.device)

    def saliency_loss(self, saliency_results, saliency_labels):
        text_map, link_map = saliency_results
        text_map_label, link_map_label, condicent_map, confidence = saliency_labels

        return self.mse(
            text_map_label.to(self.device), 
            link_map_label.to(self.device), 
            text_map, 
            link_map, 
            condicent_map.to(self.device)
        )

    def direction_loss(self, direction_results, direction_labels):
        a, p = direction_results
        a_label, p_label = direction_labels
        return self.sigma * self.ce1(p, p_label.to(self.device)) \
              +  (1 - self.sigma) * self.ce2(a, a_label.to(self.device))
    

    def recognition_loss(self, marking_results, marking_labels):
        marking_probs = marking_results
        marking_lengths = torch.tensor([marking_probs.size(0)] * marking_probs.size(1)).long()
        targets, target_lengths = marking_labels

        return self.ctc(marking_probs, targets, marking_lengths, target_lengths)
    

    def forward(self, results, labels, i = 0):
        
        saliency_results, direction_results, marking_results = results
        saliency_labels, direction_labels, marking_labels = labels

        saliency_loss =  self.saliency_loss(saliency_results, saliency_labels)
        direction_loss = self.direction_loss(direction_results, direction_labels)
        recognition_loss = self.recognition_loss(marking_results, marking_labels)

        if self.is_train:
            lamb = 1 / (1 + math.exp( (i  - self.n // 2) / (self.alpha * self.n)))
        else:
            lamb = 0.5
        
        total_loss = (1 - lamb) * saliency_loss + 0.5 * lamb * (direction_loss + recognition_loss)

        return total_loss, {
            'saliency_loss': saliency_loss.cpu().item(),
            'direction_loss': direction_loss.cpu().item(),
            'recognition_loss': recognition_loss.cpu().item()
        }
    

class SaliencyLoss(nn.Module):

    def __init__(self):
        super(SaliencyLoss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)

        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss

        return sum_loss

    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah
        loss_fn = torch.nn.MSELoss(reduction='none')

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss2 = loss_fn(p_gah, gah_label)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)

        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]