import snscrape.modules.twitter as sntwitter
import pandas as pd
import file_page_analysis
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
import plotly.express as px
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import streamlit as st
from datetime import datetime
from keras_preprocessing.sequence import pad_sequences
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

def file_page():
    st.title("Sentiment Analysis 😊😐😕😡")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("User Input for Tweet Analysis")
    st.text("Analyzing tweets of input given by the user and find sentiments within it.")
    st.text("")
    userText = st.text_input('User Input', placeholder='Input tweet  HERE')
    st.text("")

    userStartDate = st.text_input('Input Start Date HERE in Format : YYYY-MM-DD', value='2022-01-01')
    userEndDate = st.text_input('Input End Date HERE in Format : YYYY-MM-DD', value='2022-01-31')

    if userStartDate and userEndDate:
        userStarttemp = datetime.strptime(userStartDate, '%Y-%m-%d')
        userStart = userStarttemp.strftime('%Y-%m-%d')
        userEndtemp = datetime.strptime(userEndDate, '%Y-%m-%d')
        userEnd = userEndtemp.strftime('%Y-%m-%d')
    else:
        st.warning('Please enter values for both start and end dates.')

    if st.button('Predict'):
        if(userText!="" and userStart!="" and userEnd!=""):
            query = f"(from:{userText}) until:{userEnd} since:{userStart}"
            limit = 50
            
            dates = []
            users = []

            tweets = []
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                if(len(tweets) == limit):
                    break
                else: 
                    dates.append(tweet.date)
                    users.append(tweet.user.username)
                    tweets.append(tweet.rawContent)
            data = {'Date': dates, 'User': users, 'Tweet': tweets}
            file = pd.DataFrame(data)
            file.to_csv(f"{userText}.csv")
            if file is not None:
                # Read the input file
                try:
                   file_page_analysis.file_page_A(f"{userText}.csv") 
                except ValueError as e:
                    # If there's an error, show the error message and stop the app
                    st.error(str(e))
                    return


if __name__ == '__file_page__':
    file_page()
