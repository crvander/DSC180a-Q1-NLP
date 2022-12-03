import sys
import yaml
from box import Box
import pandas as pd
import time
sys.path.insert(0, 'src')
from data.make_dataset import download_data, generate_data, save_data
from train import train
from test import test
import logging

def main(args):
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('loading data-params...')
    with open('config/data-params.yml', 'r') as file: # All config will be read in module files
        data_config = Box(yaml.full_load(file))
    logging.info(data_config) # here only for logging
    
    download_data()
    df = generate_data()
    save_data(df)

    logging.info('loading training-params...')
    with open('config/train-params.yml', 'r') as file:
        train_config = Box(yaml.full_load(file))
    logging.info(train_config)
     
    start = time.time()
    trainer = train()
    end = time.time()
    logging.info('training time: ' + str(end - start))
    
    logging.info('test start...')
    test(test_lines = 50)
    return


if __name__ == '__main__':
    main(sys.argv[1:])
