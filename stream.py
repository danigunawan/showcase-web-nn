import io
import time
import uuid
import threading

from flask import Blueprint, render_template, request, send_file, Response, abort
import numpy as np
import cv2
from PIL import Image
import base64

from server_config import config
import style_transfer


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
    
    frame=np.array(list(request.data)).reshape([*camera_size,3]).astype(np.uint8)
    
    connection.push(frame)
    return ""
    """
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
    """
def add_frame(cid, data):
    connection=get_connection(cid)
    if connection == None:
        abort(404)
    camera_size=(int(config["IMAGE"]["size_x"]), int(config["IMAGE"]["size_y"]))
    frame=np.array(list(data)).reshape([*camera_size,4])[:,:,0:3].astype(np.uint8)
    
    connection.push(frame)
    


@stream_app.route("/<cid>/stream")
def stream(cid):
    connection=get_connection(cid)
    if connection == None:
        abort(404)
    #return Response(connection, mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(connection, mimetype='text/event-stream')
def gen(connection):
    while True:
        for frame in connection:
            yield frame


    

def get_connection(cid):
    return connections.get(cid, None)

def create_connection():
    model=style_transfer.transform
    conn_id=str(uuid.uuid1())
    connection=Connection(conn_id, model)
    connections[conn_id]=connection
    return connection


class Connection:
    def __init__(self, cid, model):
        self.cid=cid
        self.model_stream=ModelStream(FrameBufferQueue(), model)

    def __iter__(self):
        return self.model_stream.__iter__()
    def push(self, frame):
        self.model_stream.fbq.push(frame)
        
class ModelStream:
    def __init__(self, fbq, model):
        self.fbq=fbq
        self.model=model

    def __iter__(self):
        return self

    def __next__(self):
        frame=self.fbq.get_frame()

        im = Image.fromarray(self.model(frame))
        byte_io = io.BytesIO()
        im.save(byte_io, 'JPEG')
        byte_io.seek(0)
        #print("sending")
        #print(byte_io.read())
        #base64.b64encode(byte_io.getvalue())
        #print(base64.b64encode(byte_io.getvalue()).decode())
        return ServerSentEvent(base64.b64encode(byte_io.getvalue()).decode()).encode()
        #return (b'--frame\r\n'
        #       b'Content-Type: image/jpeg\r\n\r\n' + byte_io.read() + b'\r\n')

    

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
