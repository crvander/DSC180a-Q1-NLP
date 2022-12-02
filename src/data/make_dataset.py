import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

def generate_data(expand = 'True'):
    # subprocess.run('~/.local/bin/kaggle datasets download -p /home/xic023/DSC180A-Methodology-5/data/raw ankurzing/sentiment-analysis-for-financial-news', shell = True, stdout = subprocess.PIPE)
    # subprocess.run('~/.local/bin/kaggle datasets download -p /home/xic023/DSC180A-Methodology-5/data/raw yash612/stockmarket-sentiment-dataset', shell = True, stdout = subprocess.PIPE)
    # subprocess.run('unzip /home/xic023/DSC180A-Methodology-5/data/raw/\*.zip -d /home/xic023/DSC180A-Methodology-5/data/raw', shell = True, stdout = subprocess.PIPE)

    logging.info('loading datasets....')
    df1 = pd.read_csv('../data/raw/financial-news/all-data.csv', delimiter=',', encoding='latin-1',
                      names=['sentiment', 'text'])
    df2 = pd.read_csv('../data/raw/tweets/stock_data.csv')
    df3 = pd.read_csv('../data/raw/tweets/tweets_labelled_09042020_16072020.csv', on_bad_lines='skip', sep=';')
    # df4 = ... Dylan's api
    logging.info('datasets loaded')

    def convert_sentiment(sent):
        if sent == 'neutral':
            return 0
        elif sent == 'positive':
            return 1
        else:
            return -1

    logging.info('preprocessing...')
    df1['sentiment'] = df1['sentiment'].apply(convert_sentiment)
    df1 = df1[['text', 'sentiment']]
    df2.rename(columns={'Text': 'text', 'Sentiment': 'sentiment'}, inplace=True)
    df3.rename(columns={'Sentiment': 'sentiment', 'Text': 'text'}, inplace=True)
    if expand:                  # to concat Dylan's data
        # df = pd.concat([df1, df2, df3, df4], ignore_index=True, axis=0)
        df = pd.concat([df1, df2, df3], ignore_index=True, axis=0)
    else:
        df = pd.concat([df1, df2, df3], ignore_index=True, axis=0)
    df.rename(columns={'sentiment': 'labels'}, inplace=True)
    logging.info('preprocessing completed.')
    return df


def save_data(df, split = 0.2, random_state=42, save_path = ''):
        logging.info('train test with {} split, random state {}'.format(split, str(random_state)))
        train, test = train_test_split(data, test_size=split, random_state = random_state)
        logging.info('saving training and testing data...')
        train.to_csv(save_path + train_name, index=False)
        test.to_csv(save_path + test_name, index=False)
        logging.info('training and testing saved at {}')
        return