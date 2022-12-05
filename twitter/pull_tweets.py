from pytwitter import Api
import twitter_credentials
import pandas as pd

api = Api(
    consumer_key=twitter_credentials.CONSUMER_KEY,
    consumer_secret=twitter_credentials.CONSUMER_SECRET,
    access_token=twitter_credentials.ACCESS_TOKEN,
    access_secret=twitter_credentials.ACCESS_TOKEN_SECRET,
)

def call_stock(stock_name):
    tweets = api.search_tweets(query=stock_name, max_results=100).data
    data = pd.Series([x.text for x in tweets])
    data.to_csv("data/raw/data.csv", index=False)
