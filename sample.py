
'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 31 July 2024
'''
import cv2
import numpy as np

axialities = [0, 1, 0, 1]
postivities = [1, 1, 0, 0]

class BBox(object):
    def __init__(self, box):
        self.x1 = box["coordinates"]["x"] - box["coordinates"]["width"] / 2
        self.y1 = box["coordinates"]["y"] - box["coordinates"]["height"] / 2
        self.x2 = box["coordinates"]["x"] + box["coordinates"]["width"] / 2
        self.y2 = box["coordinates"]["y"] + box["coordinates"]["height"] / 2

        # Back up the original coordination for further resizing
        self.x1_, self.y1_, self.x2_, self.y2_ = self.x1, self.y1, self.x2, self.y2

        # Non-character markings
        if box["label"] == "#" or box["label"] == "$":
            self.text = box["label"]
            self.angle = None
            self.valid = False
        else:
            text, angle = box["label"].split("/")
            self.text = text
            self.angle = int(angle)
            self.valid = True

    # rotate the bounding box along with the image
    # update the coordinates on the image
    def rotate(self, M, angle):
        self.x1, self.y1 = M.dot(np.array([self.x1, self.y1, 1]))
        self.x2, self.y2 = M.dot(np.array([self.x2, self.y2, 1]))

        if angle == 180 or angle == 90:
            self.x1, self.x2 = self.x2, self.x1
        if angle == 270 or angle == 180:
            self.y1, self.y2 = self.y2, self.y1

        if self.angle is not None:
            self.angle = (self.angle + angle) % 360

    # output the box coordinates as a list
    def toList(self):
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    # output the resized coordinates as a list
    def toResizedList(self):
        return [int(self.x1_), int(self.y1_), int(self.x2_), int(self.y2_)]


class Sample(object):
    def __init__(self, name, img, label):
        self.name = name
        self.img = img.copy()
        self.height, self.width = img.shape[:2]
        self.boxes = []
        for box in label[0]["annotations"]:
            self.boxes.append(BBox(box))

    # draw the bounding box for preview
    def drawLabel(self):
        for box in self.boxes:
            color = (0, 0, 255) if box.text == "$" else (0, 255, 255)
            box.x1, box.y1, box.x2, box.y2 = np.array(
                [box.x1, box.y1, box.x2, box.y2]
            ).astype(int)
            cv2.rectangle(self.img, (box.x1, box.y1), (box.x2, box.y2), color, 1)

    # rotate need to be called before resized
    def rotate(self, angle=0):
        if angle == 0:
            return

        M = cv2.getRotationMatrix2D((self.width // 2, self.height // 2), -angle, 1)
        if angle == 90 or angle == 270:
            self.width, self.height = self.height, self.width
            M[0, 2] += (self.width - self.height) // 2
            M[1, 2] += (self.height - self.width) // 2

        self.img = cv2.warpAffine(self.img, M, (self.width, self.height))
        for box in self.boxes:
            box.rotate(M, angle)

    # resize the image
    # the corresponding bounding boxes are also resized
    def resize(self, tar_size):
        for box in self.boxes:
            ws, hs = (self.width, self.height)
            wt, ht = tar_size
            box.x1_ = wt / ws * box.x1
            box.y1_ = ht / hs * box.y1
            box.x2_ = wt / ws * box.x2
            box.y2_ = ht / hs * box.y2

    # output all the bounding boxes as a numpy array
    def toArray(self):
        return np.array(
            [box.toList() for box in filter(lambda box: box.valid is True, self.boxes)]
        )
    def toResizedArray(self):
        return np.array(
            [
                box.toResizedList()
                for box in filter(lambda box: box.valid is True, self.boxes)
            ]
        )

    # output the direction angle annotation as a list
    def angles(self):
        return np.array(
            [box.angle for box in filter(lambda box: box.valid is True, self.boxes)]
        )

    # output the axiality and positivity ground truth as a list
    def directions(self):
        return ([axialities[box.angle // 90] 
                 for box in filter(lambda box: box.valid is True, self.boxes)],
                 [postivities[box.angle // 90] 
                  for box in filter(lambda box: box.valid is True, self.boxes)])
    
    # output the marking contents annotation as a list
    def texts(self):
        return np.array(
            [box.text.lower().replace("-", "") 
             for box in filter(lambda box: box.valid is True, self.boxes)]
        )
