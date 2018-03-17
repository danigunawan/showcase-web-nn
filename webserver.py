from flask import request, render_template
import flask

#from stream import stream_app
import stream


app = flask.Flask(__name__)
app.register_blueprint(stream.stream_app, url_prefix='/stream')
app.config['DEBUG']=True

@app.route("/")
def index():
    return "index"


@app.route("/<nn_type>")
def client(nn_type):
    return render_template("client.html",nn_type=nn_type)



def start_flask():
    print("\n".join(map(str,app.url_map.iter_rules())))
    app.run(host= '0.0.0.0',threaded=True,port=8080)
