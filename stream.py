import io

from flask import Blueprint, render_template, request, send_file, Response
import numpy as np
import cv2
from PIL import Image
import base64

from server_config import config
import style_transfer

stream_app = Blueprint('stream_app', __name__)

@stream_app.route("/<nn_type>/send", methods=["POST"])
def index(nn_type):
    camera_size=(int(config["IMAGE"]["size_x"]), int(config["IMAGE"]["size_y"]))
    frame=np.array(list(request.data)).reshape([*camera_size,3]).astype(np.uint8)
    grey=style_transfer.transform(frame)
    #print(grey)
    im = Image.fromarray(grey)
    #output = io.BytesIO()
    #im.save(output, format='JPEG')
    #var = output.getvalue()
    byte_io = io.BytesIO()
    im.save(byte_io, 'JPEG')
    #byte_io.seek(0)
    return base64.b64encode(byte_io.getvalue())

    #return send_file(byte_io, mimetype='image/png')

    #return var

"""
def parse_camera_blob(blob):
    camera_size=(config["IMAGE"]["size_x"], config["IMAGE"]["size_y"])
    #np.zeros([])
    arr=np.empty([*camera_size,3])
    for x in range(camera_size[0]):
        for y in (camera_size[1]):
            i=x*camera_size[1]+y
            arr=arr[0
"""
