import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
from models.styletransfer import ImageTransformerNetwork
from models.faceresnet import FaceNet, nms, draw_boxes

def model():
    def decorator(f):
        model_obj = f().cuda()
        model_obj.eval()
        models_dict[model_obj.name]=model_obj
        return f
    return decorator
models_dict = {}


@model()
class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="styletransfer"
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/styletransfer_acidcrop.pth'))


    def forward(self, cuda_frame, **kwargs):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im

@model()
class GrayScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="grayscale"
    
    
    def forward(self, cuda_frame, **kwargs):
        frame = cuda_frame.data.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.stack([frame, frame, frame], axis=2)
        im = Image.fromarray(frame)
        return im


@model()
class FaceDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="facedetection"
        self.facenet = FaceNet().cuda()
        self.facenet.load_state_dict(torch.load("/hdd/Code/pytorch-face-recognition/savedir/facenet_02_it95k.pth"))
    
    
    def forward(self, cuda_frame, **kwargs):
        boxes, classes, anchors = self.facenet(cuda_frame)
        if kwargs["only_anchors"]==True:
            boxes = anchors
        final_boxes = nms(boxes, classes, threshhold = kwargs["threshhold"], use_nms = kwargs["use_nms"])
        im = draw_boxes(cuda_frame, final_boxes, border_size = 4, color = "red")
        return im


def numpy_frame_to_cuda(numpy_frame):
    tensor = torch.from_numpy(numpy_frame).cuda().permute(2,0,1).unsqueeze(0).float()
    var = Variable(tensor, requires_grad = False, volatile = True)
    return var


def cuda_var_to_image(cuda_frame):
    frame = cuda_frame.data.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    im = Image.fromarray(frame)
    return im
