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
    logging.info(args)
    if len(args) > 0:
        data_generate = args[0]
        training = args[1]
        test_target = args[2]
    else:
        data_generate = ''
        training = ''
        test_target = ''
    logging.info(data_generate, training, test_target)
    
    logging.info('loading data-params...')
    with open('config/data-params.yml', 'r') as file: # All config will be read in module files
        data_config = Box(yaml.full_load(file))
    logging.info(data_config) # here only for logging
    
    if data_generate == 'generate_data':
        download_data()
        df = generate_data()
        save_data(df)
    
    if training == 'train':
        logging.info('loading training-params...')
        with open('config/train-params.yml', 'r') as file:
            train_config = Box(yaml.full_load(file))
        logging.info(train_config)

        start = time.time()
        trainer = train()
        end = time.time()
        logging.info('training time: ' + str(end - start))
    
    if test_target == 'test':
        logging.info('test start...')
        test(test_target = 'test', test_lines = 50)
    else:
        logging.info('test run start...')
        test(test_target = 'testdata', test_lines = 3)
    return


if __name__ == '__main__':
    main(sys.argv[1:]) # command should be "python3 run.py generate_data train test, for testrun no args needed
