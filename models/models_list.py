import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image, ImageDraw
import cv2
import numpy as np
from models.styletransfer import ImageTransformerNetwork
from models.imagecaptioning import GRUAnnotator, ImageAnnotator, Lang
from models.facenet import FaceNet, nms
from collections import deque
from torchvision import transforms

from scipy.ndimage import zoom

    
models_dict = {}
def model():
    def decorator(f):
        model_obj = f().cuda()
        model_obj.eval()
        models_dict[model_obj.name] = model_obj
        return f
    return decorator


def numpy_frame_to_cuda(numpy_frame):
    tensor = torch.from_numpy(numpy_frame).cuda().permute(2,0,1).unsqueeze(0).float()
    var = Variable(tensor, requires_grad = False, volatile = True)
    return var


def cuda_var_to_numpy(cuda_frame):
    frame = cuda_frame.data.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    return frame


def cuda_var_to_image(cuda_frame):
    frame = cuda_var_to_numpy(cuda_frame)
    im = Image.fromarray(frame)
    return im


@model()
class AutomaticImageCaptioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="automaticimagecaptioning"

        lang = Lang()
        #TCN not available due to CPU ram limitations :(    (not even a gpu bottleneck =( sad)
        #self.TCN = ImageAnnotator(n_layers=18, hidden_size=256, lang=lang).cuda()
        #self.TCN.load_state_dict(torch.load("models/savedir/TCN1750k.pth"))
        #self.TCN.eval()
        
        self.GRU = GRUAnnotator(embedding_size=512, hidden_size=512, n_layers=3, lang=lang).cuda()
        self.GRU.load_state_dict(torch.load("models/savedir/GRU1750k.pth"))
        self.GRU.eval()

        trfms = [transforms.Resize((224, 224))] 
        trfms += [transforms.ToTensor()]
        trfms += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transforms = transforms.Compose(trfms)

        self.canvas = np.zeros((448,796,3), dtype=np.uint8)


    def reformat_caption(self, caption, per_row = 13):
        final = []
        while(len(caption)>0):
            if len(caption) > per_row:
                index = caption.rfind(" ", 0, per_row)
                final += [caption[0:index]+"\n"]
                caption = caption[index:]
                if caption[0] == " ":
                    caption = caption[1:]
            else:
                final += [caption[:]]
                caption = ""

        out = "".join(final)
        return out

    def draw_on_canvas(self, im, caption):
        #draw image
        im_array = np.asarray(im)
        self.canvas[0:448, 0:448, :] = im_array
        self.canvas[:, 448:, :] = 0

        #draw caption
        off_x = 455
        font = cv2.FONT_HERSHEY_SIMPLEX

        y0, dy = 100, 42
        for i, line in enumerate(caption.split('\n')):
            yyy = y0 + i*dy
            cv2.putText(self.canvas, line, (off_x, yyy), font, 1.5  , (255,255,255),1,cv2.LINE_AA)


    
    def forward(self, cuda_frame, **kwargs):
        cuda_frame = cuda_frame[:, :, 28:476, 224:672].contiguous()
        im = cuda_var_to_image(cuda_frame)
        downscaled_cuda = Variable(self.transforms(im).unsqueeze(0).cuda(), volatile=True)

        caption = self.GRU(downscaled_cuda, None, test_time=True)
        caption = self.reformat_caption(caption)

        self.draw_on_canvas(im, caption)
        return Image.fromarray(self.canvas)



@model()
class StyleAcid(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="styleacid"
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/styletransfer_acidcrop.pth'))


    def forward(self, cuda_frame, **kwargs):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im


@model()
class StyleMosaic(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="stylemosaic"
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/style_mosaic.pth'))


    def forward(self, cuda_frame, **kwargs):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im


@model()
class StylePoly(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="stylepoly"
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/style_polycrop.pth'))


    def forward(self, cuda_frame, **kwargs):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im


@model()
class StylePolyLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="stylepolylight"
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/style_polycrop_light.pth'))


    def forward(self, cuda_frame, **kwargs):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im


@model()
class StylePolyLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="stylepolyheavy"
        self.image_transformer_network = ImageTransformerNetwork().cuda()
        self.image_transformer_network.load_state_dict(torch.load('models/savedir/style_polycrop_heavy.pth'))


    def forward(self, cuda_frame, **kwargs):
        stylized_content = self.image_transformer_network(cuda_frame) * 255
        im = cuda_var_to_image(stylized_content)
        return im


@model()
class FaceDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="facedetection"
        self.facenet = FaceNet().cuda()
        self.facenet.load_state_dict(torch.load("models/savedir/facenet_pref.pth"))
        self.size = 74
        self.face_deque = deque([np.zeros((self.size, self.size, 3), dtype=np.uint8) for _ in range(24)], maxlen=24)
        self.canvas = np.zeros((536,899,3), dtype=np.uint8)
    

    def draw_boxes_on_image(self, im, boxes, border_size = 4, color="red"):
        dr = ImageDraw.Draw(im)
        for box in boxes:
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]

            for j in range(border_size):
                final_coords = [x0-j, y0-j, x1+j, y1+j]
                dr.rectangle(final_coords, outline = color)
        return im

    
    def update_face_deque(self, boxes, im):
        if len(boxes) != 0:
            for box in boxes:
                x0 = max(0, int(box[0]))
                y0 = max(0, int(box[1]))
                x1 = min(896, int(box[2]))
                y1 = min(384, int(box[3]))

                face = np.asarray(im)[y0:y1, x0:x1, :]
                face = zoom(face, (self.size/(y1-y0), self.size/(x1-x0), 1))
                self.face_deque.pop()
                self.face_deque.appendleft(face)


    def draw_on_canvas(self, im):
        im_array = np.asarray(im)
        self.canvas[0:384, 1:897, :] = im_array

        y_off = 385
        size = self.size
        for i, face in enumerate(self.face_deque):
            x = i % 12
            y = i // 12
            self.canvas[y_off+y+y*size:y_off+y+(y+1)*size, x+x*size:x+(x+1)*size :] = face

    
    def forward(self, cuda_frame, **kwargs):
        cuda_frame = cuda_frame[:, :, 60:444, :].contiguous()
        boxes, classes, anchors = self.facenet(cuda_frame)
        if kwargs["only_anchors"]==True:
            boxes = anchors
        final_boxes, final_confs = nms(boxes, classes, threshhold = kwargs["threshhold"], use_nms = kwargs["use_nms"])
        im = cuda_var_to_image(cuda_frame)
        self.update_face_deque(final_boxes, im)
        im = self.draw_boxes_on_image(im, final_boxes)
        self.draw_on_canvas(im)
        return Image.fromarray(self.canvas)


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
