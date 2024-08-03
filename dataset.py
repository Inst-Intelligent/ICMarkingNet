'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''

import numpy as np 
import json
import imageio
import cv2
from pathlib import Path
from PIL import Image
from collections import OrderedDict  

import torch
import torchvision.transforms as transforms

from sample import Sample
import utils.imgproc as imgproc
from utils.watershed import watershed
from utils.gaussian import GaussianTransformer
from utils.config import set_seed

set_seed(42)

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

def collate_fn(batch_data):

    images = torch.stack([x[0] for x in batch_data]) 
    text_map_label = torch.stack([x[1] for x in batch_data]) 
    link_map_label = torch.stack([x[2] for x in batch_data])
    condicent_map = torch.stack([x[3] for x in batch_data])
    confidence = torch.tensor([x[4] for x in batch_data])

    sample = ([x[5] for x in batch_data])

    return (
        images, 
        (text_map_label, link_map_label, condicent_map, confidence), 
        sample
    )

'''
- Arrange the image and annotations for each label
  and generate the pseudo ground truths
'''
class ItemLoader(object):

    def __init__(self, pretrained, target_size = 256, get_filename = False):

        self.pretrained = pretrained
        self.gaussianTransformer = GaussianTransformer(imgSize = target_size // 2)
        self.target_size = target_size
        self.get_filename = get_filename
        
    def setItem(self, img_path, label_path, angle = 0):
        
        self.name = img_path.name
        self.img = imageio.imread(img_path)

        self.label_loc = label_path
        self.label = json.loads(label_path.read_text())

        # Sample class take charge in the storage and transform of the bounding boxes
        # for details reference to sample.py
        sample = Sample(self.name, self.img, self.label)
        self.sample = sample
        sample.rotate(angle)
        sample.resize((self.target_size // 2, self.target_size // 2))
        self.img = sample.img
        self.labeled_bboxes = self.parse_label()

    def parse_label(self):
        labeled_bboxes = []
        for box in self.sample.boxes:
            if box.text == '$' :
                continue
            if box.text == '#' :
                length = 0
            else:
                length = len(box.text)
            labeled_bboxes.append([length, box.text, box.angle, box.toList()])

        return labeled_bboxes

    def resize_img(self, image):
        return cv2.resize(image, (self.target_size, self.target_size))

    def resize_gt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))
    
    def resize_boxes(self, image):
        pass
    
    # Calculate the confidence score for a word box
    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len
    
    # Generate pseudo ground truths
    def get_gt_label(self):
        
        self.confidence_mask = np.ones(self.img.shape[:2])
        character_bboxes = []
        words = []
        angles = []
        confidences = []

        for num_text, text, text_angle, text_bbox in self.labeled_bboxes:
 
            if num_text > 0:
                bboxes, region_scores, confidence = self.get_persudo_bboxes(num_text, 
                                                                            text_angle, 
                                                                            text_bbox)
                self.draw_confidence_mask(bboxes, confidence)
                character_bboxes.append(np.array(bboxes))
                confidences.append(confidence)
                angles.append(text_angle)
                word = '0' * num_text
                words.append(word)
            else:
                l,t,r,b=text_bbox
                self.draw_confidence_mask([np.array([[l,t],[r,t],[r,b],[l,b]],dtype=np.int32)], 0)

        region_scores = np.zeros(self.img.shape[:2], dtype=np.float32)
        affinity_scores = np.zeros(self.img.shape[:2], dtype=np.float32)
        
        # draw the pseudo text and link maps
        if len(character_bboxes) > 0:
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,character_bboxes,words,angles)
        
        # confidence score
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        
        # preprocess the image
        image = Image.fromarray(self.img)
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness = 32.0 / 255, saturation=0.5)(image)
        image = self.resize_img(np.array(image))
        image = imgproc.normalizeMeanVariance(np.array(image), mean = (0.485, 0.456, 0.406),
                                              variance = (0.229, 0.224, 0.225))
        
        # resize to the same sacle
        region_scores = self.resize_gt(region_scores)
        affinity_scores = self.resize_gt(affinity_scores)
        confidence_mask = self.resize_gt(self.confidence_mask)

        # conver the image and labels to tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores = torch.from_numpy(region_scores / 255).float()
        affinity_scores = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask = torch.from_numpy(confidence_mask).float()

        return (
            image,                   
            region_scores,           
            affinity_scores,         
            confidence_mask,         
            confidences,              
            self.sample
        )
    
    
    def get_persudo_bboxes(self, num_text, text_angle, text_bbox):
        
        # bounding box of a word
        l, t, r, b = text_bbox
        input = self.img[t:b, l:r].copy()
        # origial word box size
        input_h, input_w = input.shape[:2]
        # rotate the word box into 0 degree
        if text_angle > 0:
            input = cv2.rotate(input, 2 - text_angle // 90  + 1)
        # rotated word box size
        rotated_h, rotated_w = input.shape[:2]
        # unify the height
        scale = 64.0 / rotated_h
        input = cv2.resize(input, None, fx = scale, fy = scale)
        right_margin_res = 0 # align the width
         # resized word box size
        resized_h, resized_w = input.shape[:2]

        # padding the right border to make the width mutiples of 32
        if resized_w % 32 != 0:
            right_margin_res = (resized_w // 32 + 1) * 32 - resized_w
        
        # adaptive expanding
        # extract the main color with in the word box as background color
        bgc = np.array([np.argmax(cv2.calcHist([input], [i], None, [256], [0.0,255.0])) for i in range(3)])
        margin = 20
        # padding with the background color
        feed = np.ones((resized_h + margin * 2, resized_w + margin * 2 + right_margin_res, 3), dtype=np.uint8) * bgc
        feed = feed.astype(np.uint8)
        feed[margin:margin + resized_h, margin:margin + resized_w,:] = input
        input = feed
        
        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.float().cuda()
        
        # use pre-trained model, inference the text map of the word box
        scores, _,  = self.pretrained(img_torch) 
        region_scores = scores[0, :, :, 0].cpu().data.numpy()
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255) 
        region_scores = cv2.resize(region_scores, None, fx = 2, fy = 2)
        region_scores_color = cv2.cvtColor(region_scores, cv2.COLOR_GRAY2BGR)

        # calculate the character box using the text-map
        pseudo_boxes = watershed(region_scores_color, region_scores, low_text = 0.5)
        # calculate the confidence score of the word box
        confidence = self.get_confidence(num_text, len(pseudo_boxes))
        bboxes = []
        
        # if confidence<=0.5, split the word box equally
        # details as in the paper: "Character Region Awareness for Text Detection"
        if confidence <= 0.5:
            width = resized_w
            height = resized_h
            width_per_char = width / num_text
            for i in range(num_text):
                left = i * width_per_char
                right = (i + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)
            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5
        else:
            # remove the expansion area after text map inference
            bboxes = pseudo_boxes - np.array([margin, margin])

        bboxes = bboxes / scale
 
        # mapping the character box into its original direction
        bboxes = np.roll(bboxes, 4 - text_angle // 90, axis = 1)
        # for 90 and 270 degrees, swap the x and y coordinate
        if text_angle % 180 == 90:
            bboxes = np.roll(bboxes, 1, axis = 2)

        # calculating the character coordinate in the entire image
        expand_width = 1
        # padding the short side
        if text_angle % 180 == 0:
            relative_corner = np.matrix([[l,t - expand_width],[l,t - expand_width],[l,t + expand_width],[l,t + expand_width]])
        else:
            relative_corner = np.matrix([[l - expand_width,t],[l + expand_width,t],[l + expand_width,t],[l - expand_width,t]])
        
        if 90 <= text_angle <= 180:
            bboxes[:, :, 0] = input_w - bboxes[:, :, 0]
        if text_angle >= 180:
            bboxes[:, :, 1] = input_h - bboxes[:, :, 1]
            
        # the final character boxes
        for i in range(len(bboxes)):
            startidx = bboxes[i].sum(axis=1).argmin()
            bboxes[i] = np.roll(bboxes[i], 4 - startidx, 0)
            temp = np.matrix(bboxes[i])
            bboxes[i] = np.array(temp + relative_corner)

        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., self.img.shape[1] - 1)
        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., self.img.shape[0] - 1)
        bboxes = sorted(bboxes, key = lambda x:x[0][0] + x[0][1])
     
        return bboxes, region_scores, confidence

            
    def draw_confidence_mask(self, bboxes, confidence):
        
        for bbox in bboxes:
            cv2.fillPoly(self.confidence_mask, [np.int32(bbox)], (confidence))

'''
Dataset Class
'''
class ICData(torch.utils.data.Dataset):
    
    def __init__(self, 
                 pretrained, 
                 data_dir, 
                 label_dir, 
                 target_size=256, 
                 device=torch.device('cpu'), 
                 get_filename=False, 
                 direct_aug = False,
                 ):
        
        torch.cuda.set_device(device)

        if isinstance(data_dir,str):
            data_dir = Path(data_dir)
        if isinstance(label_dir,str):
            label_dir = Path(label_dir)

        self.pretrained = pretrained
        self.pretrained.eval()
        self.data_dir = data_dir
        self.itemLoader = ItemLoader(pretrained, target_size, get_filename)
        self.direct_aug = direct_aug

        data_list = []
        for x in data_dir.iterdir():
            if not x.suffix == '.jpg' and not x.suffix == '.jpeg':
                continue
            labelText = (label_dir / x.stem).with_suffix('.json')

            if not labelText.exists():
                continue
            data_list += [[x, labelText]]
        self.data_list = data_list
        self.load_label()
    
    # augment each image to other three directions
    def load_label(self):
        self.data = []
        for i in range(len(self.data_list)):
            for angle in ([0, 90, 180, 270] if self.direct_aug else [0]):
                self.itemLoader.setItem(*self.data_list[i], angle)
                self.data.append(self.itemLoader.get_gt_label())

    def update_label(self):
        for i in range(len(self.data)):
            if self.data[i][-1] < 1.0:
                self.itemLoader.setItem(*self.data_list[i])
                self.data[i] = self.itemLoader.get_gt_label()
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)