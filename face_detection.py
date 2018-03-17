import torch
import torch.nn.functional as F

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