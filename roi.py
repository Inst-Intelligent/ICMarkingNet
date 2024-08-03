'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align, roi_pool
from cc_torch_2 import connected_components_labeling

text_box_iou_threshold = 0.3
link_threshold = 0.40
text_threshold = 0.65
low_text =  0.7


'''
[Global Variables]
- map_ones and map_zeros are used for thresholding
  the text and link maps in RoI inference
'''
map_size = (128, 128)
map_ones = torch.ones(map_size, dtype=torch.uint8)
map_zeros = torch.zeros(map_size, dtype=torch.uint8)
char_roi_padding = torch.Tensor([0, -3, -3, 3, 3])


CHARS = '0123456789abcdefghjklmnpqrstuvwxyz-'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}



'''
- Threshold the text map and link map and get a binary mask
'''
def get_text_mask(text_map, threshold):
    global map_ones, map_zeros
    if map_ones.device != text_map.device:
       map_ones = map_ones.to(text_map.device)
    if map_zeros.device != text_map.device:
       map_zeros = map_zeros.to(text_map.device)

    return torch.where(text_map >= threshold, map_ones, map_zeros)

def get_link_mask(link_map, threshold):
    global map_ones, map_zeros
    if map_ones.device != link_map.device:
       map_ones = map_ones.to(link_map.device)
    if map_zeros.device != link_map.device:
       map_zeros = map_zeros.to(link_map.device)
    return torch.where(link_map >= threshold, map_ones, map_zeros)


'''
- Inference the word bounding box from the masks
- more_info contains annotations
- The predicted bounding boxes are matched with the the ground truth
  by IoU calculating, and thus direction and marking content labels
'''
def find_word_boxes_per_sample(text_mask, link_mask):
    labels = connected_components_labeling(text_mask + link_mask)
    predict_text_boxes = []
    for label in torch.unique(labels)[1:]:
        label_mask = (labels == label).byte()
        nonzero_indices = torch.nonzero(label_mask, as_tuple=False)
        min_indices = torch.min(nonzero_indices, dim=0).values
        max_indices = torch.max(nonzero_indices, dim=0).values + 1
        predict_text_boxes.append(torch.hstack([max_indices, min_indices]).flip(0))

    return torch.stack(predict_text_boxes).float() if len(predict_text_boxes) else None

def find_word_boxes_with_labels(text_maps, link_maps, more_info):
    words_roi_list = []
    a_hat_list = []
    p_hat_list = []
    m_hat_list = []
    m_len_list = []

    text_masks = get_text_mask(text_maps, text_threshold)
    link_masks = get_link_mask(link_maps, link_threshold)

    for i in range(text_maps.size(0)):

        pboxes  = find_word_boxes_per_sample(text_masks[i], link_masks[i])
        if pboxes is None: continue

        gboxes = torch.from_numpy(more_info[i].toResizedArray()).to(pboxes)
        text_ious = box_iou(pboxes, gboxes)
        text_match_score, text_match_index = torch.max(text_ious, dim = 1)
        text_match_filter = text_match_score > text_box_iou_threshold
        pboxes = pboxes[text_match_filter].clip(0, map_size[0])
        word_match_index = text_match_index[text_match_filter]

        axiality, posvitity = more_info[i].directions()
        markings = more_info[i].texts()

        a_hat = [axiality[k] for k in word_match_index]
        p_hat = [posvitity[k] for k in word_match_index]
        m_hat = [torch.LongTensor([CHAR2LABEL[c] for c in markings[k]]) for k in word_match_index]
        m_len = [len(markings[k]) for k in word_match_index]

        words_roi_list += [F.pad(pboxes, (1,0), 'constant', i)]

        a_hat_list += a_hat
        p_hat_list += p_hat
        m_hat_list += m_hat
        m_len_list += m_len
    
    return (
        torch.cat(words_roi_list, dim = 0), 
        torch.tensor(a_hat_list).long(), 
        torch.tensor(p_hat_list).long(), 
        torch.cat(m_hat_list, dim = 0) if len(m_hat_list) > 0 else torch.empty((0)),
        torch.tensor(m_len_list).long()
    )


'''
- Link point inference for LinkSampling
- The binaried link map is smoothed and pooled so as to remove noises,
  and keep the real locations of maxima as the link points
- The link points generate the boundary of each feature group
  alinging with features of ONE character
'''
def gaussian_kernel(kernel_size, sigma):
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return gaussian_kernel / gaussian_kernel.sum()

def create_smooth_gaussian_kernel():
    kernel_size = 5
    sigma = 1.0
    gaussian = gaussian_kernel(kernel_size, sigma)
    gaussian_filter = nn.Conv2d(1, 1, kernel_size, padding = kernel_size // 2, bias = False)
    gaussian_filter.weight.data = gaussian.view(1, 1, kernel_size, kernel_size)
    return gaussian_filter

gaussian_filter = create_smooth_gaussian_kernel()
def find_links(link_map):
    global gaussian_filter
    assert gaussian_filter is not None
    gaussian_filter.to(link_map.device)

    link_map = link_map.permute(0,2,1)
    link_map_smooth = link_map.unsqueeze(1)
    link_map_smooth = gaussian_filter(link_map_smooth)

    local_maxima = F.max_pool2d(link_map_smooth, 
                                kernel_size=9, 
                                stride=1, 
                                padding=4)
    local_maxima = (torch.logical_and(
        local_maxima == link_map_smooth, 
        link_map_smooth > 0.25)).int()
    local_maxima = local_maxima.squeeze()
    return torch.nonzero(local_maxima)

def roi_pooling(features, words_roi):

    squared_word_featuers = roi_pool(features, words_roi, (16,16), spatial_scale=0.5)
    return squared_word_featuers

def link_sampling(features, words_roi, links, axiality, postivity):
    n_words, n_links = words_roi.size(0), links.size(0)
    axiality = torch.round(F.softmax(axiality, dim = 1))
    postivity = torch.round(F.softmax(postivity, dim = 1))
    
    # remove the first column, which is the batch index
    links_ = torch.hstack((links, links[:,1:])) 

    # axiality and postivity are predicted by the Direction module
    # they decide the how to arrange and whether to rotate the sampled
    # character features
    sgn = (2 * torch.argmax(postivity, dim=1) - 1).view(-1,1)
    word_theta = torch.zeros((n_words, 6)).to(features.device)
    sin_cos = sgn * axiality

    # construct the rotation transormation matrix
    # for unifying the text feature  into one direction
    word_theta[:,0:2] = sin_cos * torch.as_tensor([1, -1]).to(sin_cos.device)
    word_theta[:,3:5] = sin_cos.flip(1) 

    ll = links_.repeat(1, n_words).view(n_links, n_words, 5) \
        * torch.as_tensor([1, 1, 1, -1, -1]).to(links_.device)
    ww = words_roi.repeat(n_links,1).view(n_links, n_words, 5) \
        * torch.as_tensor([-1, -1, -1, 1, 1]).to(words_roi.device)
    test = (ll + ww) 
    test[:,:,0] = -test[:,:,0] ** 2 + 1

    match = (test>0).all(dim=2).nonzero()

    feature_list = []
    for i in range(n_words):
        word = words_roi[i] 
        sample_index = word[0].long()
        link_indices = match[torch.nonzero(match[:,1]==i),0].view(-1)
        
        links_in_word = links[link_indices,1:]

        word_features, _ = get_word_features(features[sample_index:sample_index+1], 
                                             word,
                                             links_in_word, 
                                             axiality[i], sgn[i], word_theta[i])
        feature_list.append(word_features)

    return torch.stack(feature_list, dim  = 0)


def get_word_features(features, 
                      word_roi, 
                      links_in_word, 
                      axiality, 
                      sgn, 
                      theta,
                      num_group = 16,
                      sample_level = 2):
    global char_roi_padding
    
    # marker of verticiation (axiality = 1)
    vert = torch.argmax(axiality, dim=0)
    n_links =  links_in_word.size(0)

    # calculating the position of linkpoints
    # get x coordinate of horizontal words,
    # while get y coordinate of the vertical words
    w = torch.empty((n_links + 2), device=word_roi.device)
    w[:2] = word_roi[[2,4]] if vert else word_roi[[1,3]]
    if n_links: w[2:] = links_in_word[:,1] if vert else links_in_word[:,0]

    # remove the duplication
    w = w.unique().sort().values 
    n_links = w.size(0) - 2 
    boxes = torch.zeros((n_links + 1,5), dtype = torch.float, device = word_roi.device)
    
    # calculating the new boundaries for each character
    if vert:
        boxes[:,[1,3]] = word_roi[[1,3]]
        if n_links: boxes[:,2],boxes[:,4] = w[:-1], w[1:]
    else:
        boxes[:,[2,4]] = word_roi[[2,4]]
        if n_links: boxes[:,1],boxes[:,3] = w[:-1], w[1:]

    # flip the sequence of characters if positivity is negative
    if sgn == -1:
        boxes=boxes.flip(0)
    
    # group and subsample the character features
    if char_roi_padding.device != boxes.device:
        char_roi_padding = char_roi_padding.to(boxes.device)
    boxes += char_roi_padding
    char_features = roi_align(features, boxes, (8,8), spatial_scale=0.5, aligned=True)
    char_theta = theta.repeat(n_links + 1, 1).view(-1,2,3)

    # execute the direction unification
    char_grid = F.affine_grid(char_theta, char_features.size(), align_corners=False)
    char_features = F.grid_sample(char_features, char_grid, align_corners=False)
    char_features = F.adaptive_avg_pool2d(char_features, (8, sample_level))

    # align the word features into the sample length
    word_features = F.pad(
        char_features.permute(1,2,0,3).reshape(-1, 8, (n_links + 1) * sample_level),
        (0,((num_group - n_links - 1)) * sample_level),
        value = 0
    )

    return word_features, boxes