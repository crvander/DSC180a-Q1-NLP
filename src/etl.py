import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split


def generate_data():
    sentiment = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_1 = pd.read_csv('../data/raw/financial-news/all-data.csv', delimiter=',', encoding='latin-1',
                       names=['sentiment', 'text'])
    df_1.rename(columns={'sentiment': 'label'}, inplace=True)
    df_1.label = [sentiment[item] for item in df_1.label]

    df_2 = pd.read_csv('../data/raw/tweets/tweets_labelled_09042020_16072020.csv', on_bad_lines='skip', sep=';')
    df_2 = df_2.dropna()[['text', 'sentiment']]
    df_2.rename(columns={'sentiment': 'label'}, inplace=True)
    df_2.label = [sentiment[item] for item in df_2.label]

    df_3 = pd.read_csv('../data/raw/tweets/stock_data.csv')
    df_3.rename(columns={'Sentiment': 'label', 'Text': 'text'}, inplace=True)

    df = pd.concat([df_1, df_2, df_3])
    return df


def save_data(data):
    train, test = train_test_split(data, test_size=0.1)
    train.to_csv('../data/temp/train.csv', index=False)
    test.to_csv('../data/temp/test.csv', index=False)
    return
