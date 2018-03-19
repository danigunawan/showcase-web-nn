import configparser
import os.path
import os
def reset_config():
    global config
    config = configparser.ConfigParser()
    config["IMAGE"]={
        "size_x":128*5,
        "size_y":128*3,
        "c2s_jpeg":0.8,
        "s2c_jpeg":0.8
    }
    config["SERVER"]={
        "port":8080,
        "https":True
    }
    config["STREAM"]={
        "buffer_size":3
    }
    write_file()


def write_file():
    global config

    with open(config_path,"w") as f:
        config.write(f)
    
def check_dirs():
    os.makedirs(config_base, exist_ok=True)

def read_config():
    config.read(config_path)
    

config=configparser.ConfigParser()

config_base=""
config_path=config_base+"server.conf"

#check_dirs()
if not os.path.isfile(config_path):
    reset_config()
read_config()
