import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import models
import numpy as np
from PIL import Image, ImageDraw

class ResNeXtBlock(nn.Module):
    def __init__(self, channels, expansion = 2, cardinality = 32):
        super(ResNeXtBlock, self).__init__()

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
    

def make_anchors_and_bbox(offsets, classes, anchors_hw, height, width):
    #offsets shape [batch_size, 4A, S, S]
    #anchors shape [A, 2]
    #classes shape [batch_size, (K+1)A, S, S]
    R, C, H, W = list(classes.size())
    A, _ = list(anchors_hw.size())

    #RESHAPE OFFSETS
    offsets = offsets.view(R,-1, A*H*W).permute(0,2,1)
            
    #RESHAPE CLASSES
    classes = classes.view(R,-1, A*H*W).permute(0,2,1)
            
    #EXPAND CENTER COORDS
    x_coords = ((torch.arange(W).cuda()+0.5)/W*width).expand(H, W)
    y_coords = ((torch.arange(H).cuda()+0.5)/H*height).expand(W, H).t()
    coord_grid = torch.stack((x_coords, y_coords), dim = 0)
    coords = coord_grid.view(2,-1).t().expand(A, -1, -1)
    anch = anchors_hw.unsqueeze(1).expand(-1,H*W,-1)

    anchors_min = coords - anch/2
    anchors_max = anchors_min + anch
    anchors_min = anchors_min.view(-1,2)
    anchors_max = anchors_max.view(-1,2)
            
    anchors = Variable(torch.cat((anchors_min, anchors_max), dim = 1))
    boxes = offsets + anchors

    return boxes, classes, anchors        


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

        channels = 128
        expansion = 2
        cardinality = 16
        block_depth = 2

        res_0 = [ResNeXtBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        #upsample = nn.ConvTranspose2d(channels*expansion, channels*expansion, 3, stride=2, padding=1)
        upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        res_1 = [ResNeXtBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual = nn.Sequential(*res_0, upsample, *res_1)

    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual(x)
        x = self.regressor(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        K = 1
        A = 6
        pi = 0.0000001
        bias = np.log(K*(1-pi)/pi)
        self.prior = Parameter(torch.cuda.FloatTensor([[bias]]).expand(A, -1, -1))
        
        self.background = nn.Conv2d(256,   A, kernel_size=3, stride=1, padding=1, bias = False)
        self.foreground = nn.Conv2d(256, A*K, kernel_size=3, stride=1, padding=1, bias = True)

        channels = 128
        expansion = 2
        cardinality = 16
        block_depth = 4

        res_0 = [ResNeXtBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        #upsample = nn.ConvTranspose2d(channels*expansion, channels*expansion, 3, stride=2, padding=1)
        upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        res_1 = [ResNeXtBlock(channels, expansion, cardinality) for _ in range(block_depth)]
        self.residual = nn.Sequential(*res_0, upsample, *res_1)


    def forward(self, x):
        #x shape [batch_size, 256, H, W]
        x = self.residual(x)
        background = self.background(x) + self.prior
        foreground = self.foreground(x)
        return torch.cat((background, foreground), dim=1)
    

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules_conv3 = list(resnet.children())[:6]
        modules_conv4 = list(resnet.children())[6]
        modules_conv5 = list(resnet.children())[7]
        
        self.BN = nn.BatchNorm3d(3)

        self.conv3 = nn.Sequential(*modules_conv3)
        self.bottleneck_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias = True)
        
        self.conv4 = nn.Sequential(*modules_conv4)
        self.bottleneck_conv4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv5 = nn.Sequential(*modules_conv5)
        self.bottleneck_conv5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias = True)

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode = "bilinear")

        self.out7BN = nn.BatchNorm3d(256)
        self.out6BN = nn.BatchNorm3d(256)
        self.out5BN = nn.BatchNorm3d(256)
        self.out4BN = nn.BatchNorm3d(256)
        self.out3BN = nn.BatchNorm3d(256)

        self.prediction_head =  PredictionHead()

        self.anchors_hw3 = torch.Tensor([[16, 16],  [16/1.6, 16],
                                         [20, 20],  [20/1.6, 20],
                                         [25, 25],  [25/1.6, 25]]).cuda()
        self.anchors_hw4 = torch.Tensor([[32, 32],  [32/1.6, 32],
                                         [40, 40],  [40/1.6, 40],
                                         [51, 51],  [51/1.6, 51]]).cuda()
        self.anchors_hw5 = torch.Tensor([[64, 64],  [64/1.6, 64],
                                         [81, 81],  [81/1.6, 81],
                                         [102, 102],  [102/1.6, 102]]).cuda()
        self.anchors_hw6 = torch.Tensor([[128, 128],  [128/1.6, 128],
                                         [161, 161],  [161/1.6, 161],
                                         [203, 203],  [203/1.6, 203]]).cuda()
        self.anchors_hw7 = torch.Tensor([[256, 256],  [256/1.6, 256],
                                         [322, 322],  [322/1.6, 322],
                                         [406, 406],  [406/1.6, 406]]).cuda()
        
    def forward(self, x, phase = "train"):
        _, _, height, width = x.size()
        x = self.BN(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        
        conv3 = self.bottleneck_conv3(conv3)
        conv4 = self.bottleneck_conv4(conv4)
        conv5 = self.bottleneck_conv5(conv5)
        # FPN Feature pyramid structure described in paper
        # "Feature Pyramid Networks for Object Detection"
        # Buttom-up pathways:
        out7 = conv7
        out7 = self.out7BN(out7)
        
        out6 = conv6 + self.upsample(out7)
        out6 = self.out6BN(out6)
        
        out5 = conv5 + self.upsample(out6)
        out5 = self.out5BN(out5)
        
        out4 = conv4 + self.upsample(out5)
        out4 = self.out4BN(out4)

        out3 = conv3 + self.upsample(out4)
        out3 = self.out3BN(out3)

        offsets7, classes7 = self.prediction_head(out7)
        boxes7, classes7, anchors7 = make_anchors_and_bbox(offsets7, classes7, self.anchors_hw7, height, width)

        offsets6, classes6 = self.prediction_head(out6)
        boxes6, classes6, anchors6 = make_anchors_and_bbox(offsets6, classes6, self.anchors_hw6, height, width)

        offsets5, classes5 = self.prediction_head(out5)
        boxes5, classes5, anchors5 = make_anchors_and_bbox(offsets5, classes5, self.anchors_hw5, height, width)

        offsets4, classes4 = self.prediction_head(out4)
        boxes4, classes4, anchors4 = make_anchors_and_bbox(offsets4, classes4, self.anchors_hw4, height, width)

        offsets3, classes3 = self.prediction_head(out3)
        boxes3, classes3, anchors3 = make_anchors_and_bbox(offsets3, classes3, self.anchors_hw3, height, width)

        #concat all the predictions
        #boxes = [boxes3, boxes4, boxes5, boxes6, boxes7]
        #classes = [classes3, classes4, classes5, classes6, classes7]
        #anchors = [anchors3, anchors4, anchors5, anchors6, anchors7]
        boxes = torch.cat((boxes3, boxes4, boxes5, boxes6, boxes7), dim=1)
        classes =torch.cat((classes3, classes4, classes5, classes6, classes7), dim=1)
        anchors = torch.cat((anchors3, anchors4, anchors5, anchors6, anchors7), dim=0)
        return boxes, classes, anchors


def nms(boxes, classes, threshhold, use_nms = True):
    """ Perform non-maxima suppression on the boudning boxes
    with detection probabilities as scores
    Args:
      boxes: (tensor) bounding boxes, size [batch_size, H*W*A, 4] OR [H*W*A, 4]
      classes: (tensor) conficendes, size [batch_size, H*W*A, K+1] OR [H*W*A, K+1].
    Return:
      (tensor) the resulting bounding boxes efter nms is applied, size [X,4].
    """
    
    if len(boxes.size()) == 3:
        boxes = boxes[0]
    if len(classes.size()) == 3:
        classes = classes[0]

    classes = F.softmax(classes, dim=1)
    mask = classes > threshhold
    idx = mask[:, 1].nonzero().squeeze()
    if not len(idx.size()):
        return []
    selected_boxes = boxes.index_select(0, idx)
    selected_classes = classes.index_select(0, idx)[:,1]
    
    if(not use_nms):
        return selected_boxes
    
    confidences, indices = torch.sort(selected_classes, descending = True)
    boxes = selected_boxes[indices]

    processed_boxes = []
    while len(boxes.size()):
        highest = boxes[0:1]
        processed_boxes += [highest]
        if boxes.size(0) == 1:
            break
        below = boxes[1:]
        
        ious = jaccard(below, highest)
        mask = (ious < 0.5).expand(-1,4)
        boxes = below[mask].view(-1, 4)

    return torch.cat(processed_boxes, dim = 0)


def draw_boxes(cuda_image, list_boxes, border_size, color):
    image = cuda_image[0].data.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    
    im = Image.fromarray((image).astype(np.uint8))
    
    dr = ImageDraw.Draw(im)
    for box in list_boxes:
        #box = box[0].cpu().numpy().astype(np.int)
        box = box.int()
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]

        for j in range(border_size):
            final_coords = [x0+j, y0+j, x1-j, y1-j]
            dr.rectangle(final_coords, outline = color)
    return im


###
### NEW INTERSECTION OVER UNION (IOU) CODE FROM 
### https://github.com/amdegroot/ssd.pytorch BELOW:
###

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
