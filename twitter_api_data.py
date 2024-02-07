import tweepy
import configparser

#read config

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tweepy.OAuth1UserHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
userText = "ipl"
userStart = "2021-01-01"
userEnd = "2022-01-01"
query = tweepy.Cursor(api.search, q = userText, lang = "en", since = userStart, until = userEnd, result_type = "recent").items(2)
print(query)
# query = f"(from:{userText}) until:{userEnd} since:{userStart}"
# print(public_tweets)
