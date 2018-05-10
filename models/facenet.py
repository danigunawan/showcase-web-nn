#cross conv3-conv7 with bilinear interpolation upsampling, color jitter

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageDraw

class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion = 4, cardinality = 1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups = cardinality, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels*expansion))
        self.relu = nn.ReLU(inplace = True)
        
        
    def forward(self, x):
        res = x
        out = self.block(x)
        out = self.relu(out+res)
        return out


class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead, self).__init__()
        self.regressor = RegressionHead()
        self.classifyer = ClassificationHead()


    def forward(self, x):
        offsets = self.regressor(x)
        confidences = self.classifyer(x)
        return offsets, confidences


class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        A = 6
        self.regressor = nn.Conv2d(256, A*4, kernel_size=3, stride=1, padding=1, bias = True)

        channels = 64
        expansion = 4
        cardinality = 1
        block_depth = 2

        res_0 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        res_1 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual0 = nn.Sequential(*res_0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.residual1 = nn.Sequential(*res_1)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual0(x)
        x = self.upsample(x)
        x = self.residual1(x)
        x = self.regressor(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        A = 6
        pi = 0.001
        bias = -np.log((1-pi)/pi)
        self.prior = Parameter(torch.FloatTensor([[bias]]).expand(A, -1, -1)).contiguous()
        
        self.conf_predictions = nn.Conv2d(256,   A, kernel_size=3, stride=1, padding=1, bias = False)

        channels = 64
        expansion = 4
        cardinality = 1
        block_depth = 2

        res_0 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        res_1 = [ResidualBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual0 = nn.Sequential(*res_0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.residual1 = nn.Sequential(*res_1)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual0(x)
        x = self.upsample(x)
        x = self.residual1(x)
        return self.conf_predictions(x) + self.prior
    

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules_conv3 = list(resnet.children())[:6]
        modules_conv4 = list(resnet.children())[6]
        modules_conv5 = list(resnet.children())[7]
        
        self.input_BN = nn.BatchNorm3d(3)

        self.conv3 = nn.Sequential(*modules_conv3)
        self.bottleneck_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias = True)
        
        self.conv4 = nn.Sequential(*modules_conv4)
        self.bottleneck_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv5 = nn.Sequential(*modules_conv5)
        self.bottleneck_conv5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0, bias = True),
                                   *[ResidualBlock(128, expansion=4) for _ in range(2)])
  
        self.conv7 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0, bias = True),
                                   *[ResidualBlock(128, expansion=4) for _ in range(2)])
        self.bottleneck_conv6 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)
        self.bottleneck_conv7 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.prediction_head =  PredictionHead()

        self.anchors_wh3 = torch.Tensor([[16, 16],  [16, 16*1.5],
                                         [20, 20],  [20, 20*1.5],
                                         [25, 25],  [25, 25*1.5]]).cuda()
        self.anchors_wh4 = torch.Tensor([[32, 32],  [32, 32*1.5],
                                         [40, 40],  [40, 40*1.5],
                                         [51, 51],  [51, 51*1.5]]).cuda()
        self.anchors_wh5 = torch.Tensor([[64, 64],  [64, 64*1.5],
                                         [81, 81],  [81, 81*1.5],
                                         [102, 102],  [102, 102*1.5]]).cuda()
        self.anchors_wh6 = torch.Tensor([[128, 128],  [128, 128*1.5],
                                         [161, 161],  [161, 161*1.5],
                                         [203, 203],  [203, 203*1.5]]).cuda()
        self.anchors_wh7 = torch.Tensor([[256, 256],  [256, 256*1.5],
                                         [322, 322],  [322, 322*1.5],
                                         [406, 406],  [406, 406*1.5]]).cuda()

        self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear")
        
    def forward(self, x, phase = "train"):
        _, _, height, width = x.size()
        x = self.input_BN(x)

        conv3 = self.conv3(x)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        conv7 = self.bottleneck_conv7(conv7)
        conv6 = self.bottleneck_conv6(conv6) + self.upsampling(conv7)
        conv5 = self.bottleneck_conv5(conv5) + self.upsampling(conv6)
        conv4 = self.bottleneck_conv4(conv4) + self.upsampling(conv5)
        conv3 = self.bottleneck_conv3(conv3) + self.upsampling(conv4)

        offsets7, classes7 = self.prediction_head(conv7)
        boxes7, classes7, anchors7 = make_anchors_and_bbox(offsets7, classes7, self.anchors_wh7, height, width)
        offsets6, classes6 = self.prediction_head(conv6)
        boxes6, classes6, anchors6 = make_anchors_and_bbox(offsets6, classes6, self.anchors_wh6, height, width)
        offsets5, classes5 = self.prediction_head(conv5)
        boxes5, classes5, anchors5 = make_anchors_and_bbox(offsets5, classes5, self.anchors_wh5, height, width)
        offsets4, classes4 = self.prediction_head(conv4)
        boxes4, classes4, anchors4 = make_anchors_and_bbox(offsets4, classes4, self.anchors_wh4, height, width)
        offsets3, classes3 = self.prediction_head(conv3)
        boxes3, classes3, anchors3 = make_anchors_and_bbox(offsets3, classes3, self.anchors_wh3, height, width)

        boxes = torch.cat((boxes3, boxes4, boxes5, boxes6, boxes7), dim=1)
        classes =torch.cat((classes3, classes4, classes5, classes6, classes7), dim=1)
        anchors = torch.cat((anchors3, anchors4, anchors5, anchors6, anchors7), dim=0)
        return boxes, classes, anchors


def make_anchors_and_bbox(offsets, classes, anchors_wh, height, width):
    #offsets shape [batch_size, 4A, H, W]
    #anchors shape [A, 2]
    #classes shape [batch_size, A, H, W]
    R, A, H, W = classes.size()

    #RESHAPE OFFSETS
    offsets = offsets.view(R, 4, A*H*W).permute(0,2,1)
            
    #RESHAPE CLASSES
    classes = classes.view(R, A*H*W)
            
    #EXPAND CENTER COORDS
    x_coords = ((torch.arange(W).cuda()+0.5)/W*width).expand(H, W)
    y_coords = ((torch.arange(H).cuda()+0.5)/H*height).expand(W, H).t()
    coord_grid = torch.stack((x_coords,y_coords), dim = 2) #H-dim, W-dim, (x,y)
    coord_grid = coord_grid.expand(A,-1,-1,-1) #A-dim, H-dim, W-dim, (x,y)
    coords = coord_grid.contiguous().view(-1, 2) #AHW, 2
    anch = anchors_wh.unsqueeze(1).expand(-1,H*W,-1).contiguous().view(-1, 2) #AHW, 2

    anchors_min = coords - anch/2
    anchors_max = anchors_min + anch
            
    anchors = Variable(torch.cat((anchors_min, anchors_max), dim = 1), requires_grad = False)
    boxes = offsets + anchors

    return boxes, classes, anchors


def nms(boxes, classes, threshhold, use_nms = True):
    """ Perform non-maxima suppression on the boudning boxes
    with detection probabilities as scores
    Args:
      boxes: (tensor) bounding boxes, size [batch_size, H*W*A, 4] OR [H*W*A, 4]
      classes: (tensor) conficendes, size [batch_size, H*W*A] OR [H*W*A].
    Return:
      (tensor) the resulting bounding boxes efter nms is applied, size [X,4].
    """
    if len(boxes.size()) == 3:
        boxes = boxes[0]
    if len(classes.size()) == 2:
        classes = classes[0]

    classes = F.sigmoid(classes)
    mask = classes > threshhold
    idx = mask.nonzero().squeeze()
    if not len(idx.size()):
        return [], []
    selected_boxes = boxes.index_select(0, idx)
    selected_classes = classes.index_select(0, idx)
    
    if(not use_nms):
        return selected_boxes, selected_classes
    
    confidences, indices = torch.sort(selected_classes, descending = True)
    boxes = selected_boxes[indices]
    confs = selected_classes[indices]

    processed_boxes = []
    processed_confs = []
    while len(boxes.size()):
        highest = boxes[0:1]
        highest_conf = confs[0:1]
        processed_boxes += [highest]
        processed_confs += [highest_conf]
        if boxes.size(0) == 1:
            break
        below = boxes[1:]
        below_conf = confs[1:]
        
        ious = jaccard(below, highest)
        mask = (ious < 0.5)
        confs = below_conf[mask.squeeze()]
        mask = mask.expand(-1,4)
        boxes = below[mask].view(-1, 4)
        
    return torch.cat(processed_boxes, dim = 0), torch.cat(processed_confs, dim = 0)


###I CANT FIGURE OUT HOW TO FIX MY OWN IOU FUNCTION,
###IT IS CORRECT EXCEPT FOR SOME EXAMPLES WHICH IT GIVES IOU>1,
###MAJOIRTY OF EXAMPLES ARE CORRECT THOUGH, FROM NOW ON USING
### NEW IOU CODE FROM https://github.com/amdegroot/ssd.pytorch

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
