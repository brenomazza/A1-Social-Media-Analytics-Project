# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:34:07 2021

@author: breno
"""

# Twitter data analysis task starter.
import html
import json
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

# First collect the data in json-file; specify file name here (adjust the number as queried)
fjson = 'raw_tweet_data_2500.json'

# read json file with tweets data
# https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
with open(fjson) as file:
    data = json.load(file)
len(data)

# tweet data record example: as documented for the Twitter API
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
data[0]

# create pandas dataframe from tweet text content
# https://stackoverflow.com/a/43175477
df_tweets = pd.DataFrame([t['full_text'] for t in data], columns=['text'])
print(df_tweets)

# add selected columns from tweet data fields
df_tweets['retweets'] = [t['retweet_count'] for t in data]
df_tweets['favorites'] = [t['favorite_count'] for t in data]
df_tweets['user'] = [t['user']['screen_name'] for t in data]
print(df_tweets)

# text cleaning function: see prior class modules
stop_words = set(stopwords.words('english'))

# strictly speaking, this is a closure: uses a wider-scope variable stop_words
# (disregard this note if you are a Python beginner)
def text_cleanup(s):
    s_unesc = html.unescape(re.sub(r"http\S+", "", re.sub('\n+', ' ', s)))
    s_noemoji = s_unesc.encode('ascii', 'ignore').decode('ascii')
    # normalize to lowercase and tokenize
    wt = word_tokenize(s_noemoji.lower())
    
    # filter word-tokens
    wt_filt = [w for w in wt if (w not in stop_words) and (w not in string.punctuation) and (w.isalnum())]
    
    # return clean string
    return ' '.join(wt_filt)

# add clean text column
# NOTE: apply in pandas applies a function to each element of the selected column
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
df_tweets['text_clean'] = df_tweets['text'].apply(text_cleanup)
print(df_tweets)

# sentiment analysis
def sentim_polarity(s):
    return TextBlob(s).sentiment.polarity

def sentim_subject(s):
    return TextBlob(s).sentiment.subjectivity

df_tweets['polarity'] = df_tweets['text_clean'].apply(sentim_polarity)
df_tweets['subjectivity'] = df_tweets['text_clean'].apply(sentim_subject)
print(df_tweets)

# define the list of brands to analyze, consistent with the search topic
#  for which the tweets were collected

wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
raw_data = wiki_table[0]
raw_data.to_excel('SP500_constituents.xlsx')
indexNames = raw_data[(raw_data['Date first added'] > '2013-01-01')].index
raw_data.drop(indexNames, inplace =True)

SP500 = raw_data['Security'].values.tolist()

for i in range(len(SP500)):

    SP500[i] = SP500[i].lower()

brands = SP500

# start a brand comparison dataframe
df_brands = pd.DataFrame(brands, columns=['brand'])
print(df_brands)

# example: tweet subset mentioning a given brand
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html
#df_tweets[df_tweets['text_clean'].str.contains("burbery")]

# function to compute average sentiment of tweets mentioning a given brand
def brand_sentiment(b):
    return df_tweets[df_tweets['text_clean'].str.contains(b)]['polarity'].mean()
# brand sentiment comparison
df_brands['average_sentiment'] = df_brands['brand'].apply(brand_sentiment)
print(df_brands)

# sentiment of stocks
stock_sentiment = df_brands.sort_values(by='average_sentiment', ascending=False).head(56)

# highest sentiment tweets
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
highest_sentiment = df_tweets.sort_values(by='polarity', ascending=False).head(5)    #[['text', 'polarity']]
lowest_sentiment = df_tweets.sort_values(by='polarity', ascending=True).head(5)
highest_sentiment_table = highest_sentiment[['text_clean', 'polarity', 'subjectivity', 'retweets', 'favorites']]
lowest_sentiment_table = lowest_sentiment[['text_clean', 'polarity', 'subjectivity', 'retweets', 'favorites']]

# most retweeted content
most_retweeted = df_tweets.sort_values(by='retweets', ascending=False).head(10)
a3 = most_retweeted[['text_clean', 'retweets', 'favorites', 'polarity', 'subjectivity']]

# combine all text for a specific brand
def brand_all_text(b):
    # https://stackoverflow.com/a/51871650
    return ' '.join(df_tweets[df_tweets['text_clean'].str.contains(b)]['text_clean'])

# most common twet content keywords for a specific brand
# https://amueller.github.io/word_cloud/auto_examples/single_word.html#sphx-glr-auto-examples-single-word-py
# https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
a = 'tesla'
wc = WordCloud(width=1200, height=800, max_font_size=110, collocations=False).generate(brand_all_text(a))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()

# for alternative visualizations, extract the keyword counts
kwords = WordCloud().process_text(brand_all_text(a))
print(kwords)

# transform that dictionary into a pandas DataFrame
df_kwords = pd.DataFrame(list(kwords.items()), columns=['keyword', 'count']).set_index('keyword')
print(df_kwords)

# plot a bar chart with the top keywords
df_kwords.sort_values(by='count', ascending=False).head(20).plot.bar()
