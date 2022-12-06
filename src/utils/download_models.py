import requests
import yaml
from box import Box
import logging
import os
import gdown
import subprocess

with open('config/model_config.yml', 'r') as file:
    data_config = Box(yaml.full_load(file))
    

def download_models():
    destination = data_config.outpath
    models = data_config.models
    model_folder_path = "{}/{}".format(os.getcwd(), destination)
    model_folder_list = os.listdir(model_folder_path)
    
    logging.info("downloading to {}: ".format(model_folder_path))
    for (key, value) in models.items():
        file_id = value
        file_name = key
        if file_name.rstrip('.zip') not in model_folder_list:
            logging.info("downloading from google shared id: {}".format(file_id))
            logging.info("saving model as name: {}".format(file_name))
            logging.info("saving path: {}/{}".format(model_folder_path, file_name))
            output = "{}/{}".format(model_folder_path, file_name)
            gdown.download(id=file_id, output=output, quiet=False)
     
    subprocess.run('unzip {}/\*.zip -d {}'.format(destination, destination), shell = True, stdout = subprocess.PIPE)
    subprocess.run('rm {}/*.zip'.format(model_folder_path),shell = True)
    

