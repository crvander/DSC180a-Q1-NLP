import sys
import yaml
import pandas as pd

sys.path.insert(0, 'src')
from data.make_dataset import generate_data, save_data
from train import train
from test import test


def main():
    args = sys.argv[1:]
    if args == 'train':
        print('loading data-params...')
        with open('config/data-params.yml', 'r') as file:
            data_config = yaml.safe_load_all(file)
        df = generate_data(expand=data_config.expand)

        save_data(df, split=data_config.split, random_state=data_config.random_state, save_path=data_config.save_path),

        print('loading training-params...')
        with open('config/train-params.yml', 'r') as file:
            train_config = yaml.safe_load_all(file)

        trainer = train(path=train_config.input_path)

        # test()

    elif args == 'testrun':
        print('test run')
    else:
        print("argument can only be train or testrun")
    return


if __name__ == '__main__':
    main()
