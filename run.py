import sys
import yaml
import pandas as pd
from box import Box

sys.path.insert(0, 'src')
from data.make_dataset import generate_data, save_data
from train import train
from test import test
import logging

def main(args):
    logging.info('loading data-params...')
    with open('config/data-params.yml', 'r') as file:
        data_config = Box(yaml.full_load(file))
    logging.info(data_config)
    df = generate_data(expand=data_config.expand)

    save_data(df, split=data_config.split, random_state=data_config.random_state, save_path=data_config.save_path, train_name = data_config.train_name, test_name = data_config.test_name),

    logging.info('loading training-params...')
    with open('config/train-params.yml', 'r') as file:
        train_config = Box(yaml.full_load(file))
    logging.info(train_config)

#    trainer = train(path=train_config.input_path)

    #test()
    return


if __name__ == '__main__':
    main(sys.argv[1:])
