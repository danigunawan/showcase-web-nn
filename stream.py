from flask import Blueprint, render_template

stream_app = Blueprint('stream_app', __name__)

@stream_app.route("/<nn_type>/send")
def index(nn_type):
    return nn_type
