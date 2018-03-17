import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np


def cuda_var_to_image(cuda_frame):
    frame = cuda_frame.data.unqueeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    im = Image.fromarray(frame)
    return im


class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.image_transformer_network = torch.load("models/savedir/model_2_acidcrop_it90k")

    
    def forward(self, cuda_frame):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im


class GrayScale(nn.Module):
    def __init__(self):
        super(GrayScale, self).__init__()
    
    
    def forward(self, cuda_frame):
        frame = cuda_frame.data.unqueeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.stack([frame, frame, frame], axis=2)
        im = Image.fromarray(frame)
        return im


def create_model_dict():
    style_transfer = StyleTransfer().cuda()
    style_transfer.eval()
    grayscale = GrayScale().cuda()
    grayscale.eval()

    return {"style transfer" : style_transfer, "grayscale" : grayscale}


model_dict = create_model_dict()
