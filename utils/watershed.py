'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''

import cv2
import numpy as np
import math
import Polygon as plg
from PIL import Image

def getDetCharBoxes_core(textmap, text_threshold=0.5, low_text=0.4):
    # prepare data
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def watershed(oriimage, image, low_text=0.6, viz=False):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    ret, binary = cv2.threshold(gray, 0.3 * np.max(gray), 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3) 
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    sure_bg = np.uint8(mb)

    ret, sure_fg = cv2.threshold(gray, low_text * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg) 
    unknown = cv2.subtract(sure_bg, surface_fg)

    ret, markers = cv2.connectedComponents(surface_fg)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg, connectivity=4)

    markers = labels.copy() + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(oriimage, markers=markers)
    oriimage[markers == -1] = [0, 0, 255]
    
    color_markers = np.uint8(markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    for i in range(2, np.max(markers) + 1):
        markers2 = np.zeros(markers.shape,dtype=np.uint8)
        markers2[markers==i]=255
        markers2 = cv2.dilate(markers2, kernel, iterations=3)
        np_contours = np.roll(np.array(np.where(markers2 == 255)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)