

import sys
import json
import pandas as pd

sys.path.insert(0, 'src')
from etl import generate_data, save_data

def main(targets):

    # data_config = json.load(open('config/data-params.json'))

    # if 'testdata' in targets:

        # data = generate_data()
        # save_data(data)
    return
        


if __name__ == '__main__':

    # targets = sys.argv[1:]
    targets = 'testdata'
    main(targets)
