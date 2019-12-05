import io
import time
import uuid
import threading
import copy
import os

from flask import Blueprint, render_template, request, send_file, Response, abort
import numpy as np
import cv2
from PIL import Image
import base64

from server_config import config
from models.models_list import models_dict, numpy_frame_to_cuda


stream_app = Blueprint('stream_app', __name__)

@stream_app.route("/<cid>/push", methods=["POST"])
def index(cid):
    """
    t = threading.Thread(target=add_frame, args=(cid,request.data))
    t.start()
    return ""
    """
    connection=get_connection(cid)
    if connection == None:
        abort(404)
    
    image = Image.open(io.BytesIO(base64.b64decode(request.data)))

    #frame=np.array(list(request.data)).reshape([*camera_size,3]).astype(np.uint8)
    #frame=np.array(list(request.data)).reshape([*camera_size,3]).astype(np.uint8)
    
    connection.push(np.array(image))
    return ""

@stream_app.route("/<cid>/stream")
def stream(cid):
    connection=get_connection(cid)
    if connection == None:
        abort(404)
    #return Response(connection, mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(connection, mimetype='text/event-stream')

@stream_app.route("/<cid>/setconf/<section>/<key>", methods=["POST"])
def set_conf(cid, section,key):
    connection=get_connection(cid)
    if connection == None:
        abort(404)
    #return Response(connection, mimetype='multipart/x-mixed-replace; boundary=frame')
    connection.set(section, key, request.data.decode())
    return "1"

@stream_app.route("/<cid>/setmodel/<model>", methods=["POST","GET"])
def set_model(cid, model):
    connection=get_connection(cid)
    if connection == None:
        abort(404)
    if not model in models_dict:
        abort(404)

    connection.model_stream.model=models_dict[model]
    return "1"

def gen(connection):
    while True:
        for frame in connection:
            yield frame


    

def get_connection(cid):
    return connections.get(cid, None)

def create_connection(model):
    conn_id=str(uuid.uuid1())
    connection=Connection(conn_id, model)
    connections[conn_id]=connection
    return connection


class Connection:
    def __init__(self, cid, model):
        self.cid=cid
        self.model_stream=ModelStream(FrameBufferQueue(), model, self)
        #self.config=copy.deepcopy(config._sections["NNPARAMS"])
        self.config={}
        for section in ["IMAGE", "NNPARAMS", "STREAM"]:
            self.config[section]={}
            for k, v in config._sections[section].items():
                if v in ["True","False"]:
                    val=v=="True"
                else:
                    val=float(v)
                self.config[section][k]=val
            
    def set(self,section, key, value):
        value=value
        section=section

        if isinstance(self.config[section][key], float):
            self.config[section][key]=float(value)
            return
        elif isinstance(self.config[section][key], bool):
            self.config[section][key]=bool(int(value))
            return

        self.config[section][key]=value

    def __iter__(self):
        return self.model_stream.__iter__()
    def push(self, frame):
        self.model_stream.fbq.push(frame)
        
class ModelStream:
    def __init__(self, fbq, model, connection):
        self.fbq=fbq
        self.model=model
        self.connection=connection

    def __iter__(self):
        return self

    def __next__(self):
        frame=self.fbq.get_frame()
        cuda_frame=numpy_frame_to_cuda(frame)

        #im = Image.fromarray(self.model(cuda_frame))
        im = self.model(cuda_frame, **self.connection.config["NNPARAMS"])
        byte_io = io.BytesIO()
        im.save(byte_io, 'JPEG', quality=int(self.connection.config["IMAGE"]["s2c_jpeg"]*100))
        byte_io.seek(0)
        return ServerSentEvent(base64.b64encode(byte_io.getvalue()).decode()).encode()

    

class FrameBufferQueue:
    def __init__(self):
        self.queue=[]

    def get_frame(self):
        while True:
            if len(self.queue) > 0:
                return self.queue.pop(0)
            time.sleep(0.01)
            
    def push(self, frame):
        self.queue.append(frame)

class ServerSentEvent(object):

    def __init__(self, data):
        self.data = data
        self.event = None
        self.id = None
        self.desc_map = {
            self.data : "data",
            self.event : "event",
            self.id : "id"
        }

    def encode(self):
        if not self.data:
            return ""
        lines = ["%s: %s" % (v, k) 
                 for k, v in self.desc_map.items() if k]
        
        return "%s\n\n" % "\n".join(lines)

connections={}

camera_size=(int(config["IMAGE"]["size_x"]), int(config["IMAGE"]["size_y"]))
jpeg_quality=int(100*float(config["IMAGE"]["s2c_jpeg"]))