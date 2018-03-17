import configparser
import os.path
import os
def reset_config():
    global config
    config = configparser.ConfigParser()
    config["IMAGE"]={
        "size_x":256,
        "size_y":256
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
