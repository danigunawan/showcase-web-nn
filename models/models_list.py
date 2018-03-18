import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np
from models.styletransfer import ImageTransformerNetwork

def model():
    def decorator(f):
        model_obj = f().cuda()
        model_obj.eval()
        models_dict[model_obj.name]=model_obj
        return f
    return decorator
models_dict = {}

def cuda_var_to_image(cuda_frame):
    frame = cuda_frame.data.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    im = Image.fromarray(frame)
    return im

@model()
class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="styletransfer"
        #self.image_transformer_network = torch.load("models/savedir/model_2_acidcrop_it90k.pt")
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/styletransfer_acidcrop.pth'))


#styletransfer_acidcrop.pth
    
    def forward(self, cuda_frame):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im

@model()
class GrayScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="greyscale"
    
    
    def forward(self, cuda_frame):
        frame = cuda_frame.data.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.stack([frame, frame, frame], axis=2)
        im = Image.fromarray(frame)
        return im



"""
def create_model_dict():
    style_transfer = StyleTransfer().cuda()
    style_transfer.eval()
    grayscale = GrayScale().cuda()
    grayscale.eval()

    return {"style transfer" : style_transfer, "grayscale" : grayscale}
"""


#model_dict = create_model_dict()


def numpy_frame_to_cuda(numpy_frame):
    tensor = torch.from_numpy(numpy_frame).cuda().permute(2,0,1).unsqueeze(0).float()
    var = Variable(tensor, requires_grad = False, volatile = True)
    return var
