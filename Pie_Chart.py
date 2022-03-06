# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:33:16 2022

@author: Karem Velez
"""
from textblob import TextBlob
import sys
import tweepy
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import re

import string
import seaborn as sns
import plotly.express as px
import twython



nltk.download('vader_lexicon')
nltk.download('punkt')

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt

def percentage(part,whole):
    return 100 * float(part)/float(whole) 

def df_pie_chart(user):
   
    positive  = 0
    negative = 0
    neutral = 0
    polarity = 0   

    Tweets = []
    neutral_list = []
    negative_list = []
    positive_list = []
    source = []
    follower = []
    screen_name = []
    in_reply_to_status_id = []
    in_reply_to_screen_name = []
    location = []
    friends_count = []
    statuses_count = []
    created_at = []
    created_at_us = []
    status_id = []
    reply = []
    retweet_count = []
    favorite_count = []
    hashtags = []
    user_mentions = []
    
    for tweet in user:
        Tweets.append(tweet.full_text)
        status_id.append(tweet.id_str)
        created_at.append(tweet.created_at)
        source.append(tweet.source)
        in_reply_to_status_id.append(tweet.in_reply_to_status_id_str)
        in_reply_to_screen_name.append(tweet.in_reply_to_screen_name)
        retweet_count.append(tweet.retweet_count)
        favorite_count.append(tweet.favorite_count)
        location.append(tweet.user.location)
        follower.append(tweet.user.followers_count)
        screen_name.append(tweet.user.screen_name)
        friends_count.append(tweet.user.friends_count)
        statuses_count.append(tweet.user.statuses_count)
        created_at_us.append(tweet.user.created_at)
        analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)

        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        polarity += analysis.sentiment.polarity
     
        if neg > pos:
         negative_list.append(tweet.text)
         negative += 1
        elif pos > neg:
         positive_list.append(tweet.text)
         positive += 1
        elif pos == neg:
         neutral_list.append(tweet.text)
         neutral += 1

         positive = percentage(positive)
         negative = percentage(negative)
         neutral = percentage(neutral)
         positive = format(positive, '.1f')
         negative = format(negative, '.1f')
         neutral = format(neutral, '.1f')

         labels = ['Positivo ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negativo ['+str(negative)+'%]']
         sizes = [positive, neutral, negative]
         colors = ['yellowgreen', 'blue','red']
        patches, texts = plt.pie(sizes,colors=colors, startangle=90)
        plt.style.use('default')
        plt.legend(labels)
        plt.title("Análisis de Sentimiento para la keyword=  "+user+"" )
        plt.axis('equal')
        plt.show()
        