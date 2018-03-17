from flask import Blueprint, render_template, request
import numpy as np
import cv2

from server_config import config
import style_transfer

stream_app = Blueprint('stream_app', __name__)

@stream_app.route("/<nn_type>/send", methods=["POST"])
def index(nn_type):
    camera_size=(int(config["IMAGE"]["size_x"]), int(config["IMAGE"]["size_y"]))
    frame=np.array(list(request.data)).reshape([*camera_size,3]).astype(np.uint8)
    gray=style_transfer.transform(frame)
    global win
    if not win:
        #win=True
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    
    
    return nn_type

win=False

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

        
