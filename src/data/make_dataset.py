import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    #subprocess.run('~/.local/bin/kaggle datasets download -p /home/xic023/DSC180A-Methodology-5/data/raw ankurzing/sentiment-analysis-for-financial-news', shell = True, stdout = subprocess.PIPE)
    #subprocess.run('~/.local/bin/kaggle datasets download -p /home/xic023/DSC180A-Methodology-5/data/raw yash612/stockmarket-sentiment-dataset', shell = True, stdout = subprocess.PIPE)
    #subprocess.run('unzip /home/xic023/DSC180A-Methodology-5/data/raw/\*.zip -d /home/xic023/DSC180A-Methodology-5/data/raw', shell = True, stdout = subprocess.PIPE)
    
    df1 = pd.read_csv('/home/xic023/DSC180A-Methodology-5/data/raw/all-data.csv', delimiter = ',', encoding = 'latin-1', names = ['sentiment', 'text'])
    df2 = pd.read_csv('/home/xic023/DSC180A-Methodology-5/data/raw/stock_data.csv')
    def convert_sentiment(sent):
        if sent == 'neutral':
            return 0
        elif sent == 'positive':
            return 1
        else:
            return -1
    df1['sentiment'] = df1['sentiment'].apply(convert_sentiment)
    df1 = df1[['text', 'sentiment']]
    df2.rename(columns = {'Text': 'text', 'Sentiment': 'sentiment'}, inplace = True)
    df = pd.concat([df1, df2], ignore_index = True, axis = 0)
    df.rename(columns = {'sentiment': 'labels'}, inplace = True)
    train, test = train_test_split(df, test_size = 0.2)
    train.to_csv('/home/xic023/DSC180A-Methodology-5/data/out/train.csv', index = False)
    test.to_csv('/home/xic023/DSC180A-Methodology-5/data/out/test.csv', index = False)
    
if __name__ == '__main__':
    main()
