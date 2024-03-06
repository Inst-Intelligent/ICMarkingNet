import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align, roi_pool
from cc_torch import connected_components_labeling

# For Binarization
map_size = (128, 128)
map_ones = torch.ones(map_size, dtype=torch.uint8)
map_zeros = torch.zeros(map_size, dtype=torch.uint8)
char_roi_padding = torch.Tensor([0, -3, -3, 3, 3])
 
text_box_iou_threshold = 0.3 
link_threshold = 0.40
text_threshold = 0.65
low_text =  0.7

flag = 0

max_word_size = 16

CHARS = '0123456789abcdefghjklmnopqrstuvwxyz-'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

def init(device):
    global map_ones, map_zeros, char_roi_padding
    map_ones = map_ones.to(device)
    map_zeros = map_zeros.to(device)
    char_roi_padding = char_roi_padding.to(device)

def gaussian_kernel(kernel_size, sigma):

    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return gaussian_kernel / gaussian_kernel.sum()


# text_map : Tensor
# threshold : float
def get_text_mask(text_map, threshold):

    return torch.where(text_map >= threshold, map_ones, map_zeros)

# link_map : Tensor
# threshold : float
def get_link_mask(link_map, threshold):

    return torch.where(link_map >= threshold, map_ones, map_zeros)


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
    global flag

    words_roi_list = []
    a_hat_list = []
    p_hat_list = []
    m_hat_list = []
    m_len_list = []

    text_masks = get_text_mask(text_maps, text_threshold)
    link_masks = get_link_mask(link_maps, link_threshold)

    # Per sample
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
        torch.LongTensor(a_hat_list), 
        torch.LongTensor(p_hat_list), 
        torch.cat(m_hat_list, dim = 0) if len(m_hat_list) > 0 else torch.empty((0)),
        torch.LongTensor(m_len_list)
    )



# link_map : Tensor
# Return -> Positions [N * 3] (N -> all links in a batch)
gaussian_filter = None
def create_smooth_gaussian_kernel():

    global gaussian_filter
    # Guassian filter
    kernel_size = 5
    sigma = 1.0

    gaussian = gaussian_kernel(kernel_size, sigma)

    gaussian_filter = nn.Conv2d(1, 1, kernel_size, padding = kernel_size // 2, bias = False)
    gaussian_filter.weight.data = gaussian.view(1, 1, kernel_size, kernel_size)

create_smooth_gaussian_kernel()

def find_links(link_map):

    assert gaussian_filter is not None
    gaussian_filter.to(link_map.device)

    link_map = link_map.permute(0,2,1)

    # Find all locations of maxima
    link_map_smooth = link_map.unsqueeze(1)
    link_map_smooth = gaussian_filter(link_map_smooth)
    window_size = 9
    local_maxima = F.max_pool2d(link_map_smooth, window_size, stride=1, padding=4)
    
    local_maxima = (torch.logical_and(local_maxima == link_map_smooth, link_map_smooth > 0.25)).int()
    local_maxima = local_maxima.squeeze()
    
    return torch.nonzero(local_maxima)


def roiPooling(features, words_roi):

    squared_word_featuers = roi_pool(features, words_roi, (16,16), spatial_scale=0.5)
    return squared_word_featuers

def linkSampling(features, words_roi, links, axiality, postivity):

    n_words, n_links = words_roi.size(0), links.size(0)
    
    axiality = torch.round(F.softmax(axiality, dim = 1))
    postivity = torch.round(F.softmax(postivity, dim = 1))
    
    links_ = torch.hstack((links, links[:,1:]))

    # +1 for 0/90 deg, -1 for 180/720 deg
    sgn = (2 * torch.argmax(postivity, dim=1) - 1).view(-1,1)
    word_theta = torch.zeros((n_words, 6)).to(features.device)

    # Transform Matrice: [cosθ' -sinθ' 0 sinθ' cosθ' 0 ]
    sin_cos = sgn * axiality
    word_theta[:,0:2] = sin_cos * torch.as_tensor([1, -1]).to(sin_cos.device)#.flip(1) 
    word_theta[:,3:5] = sin_cos.flip(1) 
    
    ll = links_.repeat(1, n_words).view(n_links, n_words, 5) * torch.as_tensor([1, 1, 1, -1, -1]).to(links_.device)
    ww = words_roi.repeat(n_links,1).view(n_links, n_words, 5) * torch.as_tensor([-1, -1, -1, 1, 1]).to(words_roi.device)
    test = (ll + ww) 
    test[:,:,0] = -test[:,:,0] ** 2 + 1

    match = (test>0).all(dim=2).nonzero()
    
    feature_list = []
    for i in range(n_words):
        word = words_roi[i] # (sample_index,x1,y1,x2,y2) -> (sample_index,x1,x2,y1,y2)
        sample_index = word[0].long()
        link_indices = match[torch.nonzero(match[:,1]==i),0].view(-1)
        
        links_in_word = links[link_indices,1:]

        word_features, boxes_in_word = get_word_features(features[sample_index:sample_index+1], 
                                          word, links_in_word, 
                                          axiality[i], sgn[i], word_theta[i])
        
        feature_list.append(word_features)
    
    return torch.stack(feature_list, dim  = 0)


def get_word_features(features, word_roi, links_in_word, axiality, sgn, theta):

    vert = torch.argmax(axiality, dim=0) # |cos(θ)| = 0 -> vertical
    n_links =  links_in_word.size(0)

    w = torch.empty((n_links + 2), device=word_roi.device)
    w[:2] = word_roi[[2,4]] if vert else word_roi[[1,3]]
    if n_links: w[2:] = links_in_word[:,1] if vert else links_in_word[:,0]

    w = w.unique().sort().values # remove the duplication
    n_links = w.size(0) - 2 
    boxes = torch.zeros((n_links + 1,5), dtype = torch.float, device = word_roi.device)
    
    if vert:
        boxes[:,[1,3]] = word_roi[[1,3]]
        if n_links: boxes[:,2],boxes[:,4] = w[:-1], w[1:]
    else:
        boxes[:,[2,4]] = word_roi[[2,4]]
        if n_links: boxes[:,1],boxes[:,3] = w[:-1], w[1:]

    if sgn == -1:
        boxes=boxes.flip(0)

    boxes += char_roi_padding

    size = (8,8)
    
    char_features = roi_align(features, boxes, size, spatial_scale=0.5, aligned=True)
    char_theta = theta.repeat(n_links + 1, 1).view(-1,2,3)
    
    unit_width = 2 # k for sampling

    char_grid = F.affine_grid(char_theta, char_features.size(), align_corners=False)
    char_features = F.grid_sample(char_features, char_grid, align_corners=False)
    char_features = F.adaptive_avg_pool2d(char_features, (8, unit_width))

    word_features = F.pad(
        char_features.permute(1,2,0,3).reshape(-1, 8, (n_links + 1) * unit_width),
        (0,((max_word_size - n_links - 1)) * unit_width),
        value = 0
    )

    return word_features, boxes