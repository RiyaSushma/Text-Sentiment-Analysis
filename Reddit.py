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
import praw
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

user_agent = "Text Sentiment Analysis 1.0 by /u/student"

reddit = praw.Reddit(
    client_id = "yPKO97zc84vyDNtEN8x64A",
    client_secret = "FhytIv7N3-QO_qexvr9oB6vn0cBceg",
    user_agent = user_agent
)


def file_page():
    st.title("Sentiment Analysis 😊😐😕😡")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("User Input for Tweet Analysis")
    st.text("Analyzing tweets of input given by the user and find sentiments within it.")
    st.text("")
    userText = st.text_input('User Input', placeholder='Input tweet  HERE')
    st.text("")

    start_time = st.text_input('Input Start Date HERE in Format : YYYY-MM-DD', value='2022-01-01')
    end_time = st.text_input('Input End Date HERE in Format : YYYY-MM-DD', value='2022-01-31')

    if start_time and end_time:
        start_date = datetime.strptime(start_time, '%Y-%m-%d')
        end_date = datetime.strptime(end_time, '%Y-%m-%d')
    else:
        st.warning('Please enter values for both start and end dates.')

    if st.button('Predict'):
        if(userText!="" and start_date!="" and end_date!=""):
            limit = 50
            headlines = set()
            head = []
            # start_date = datetime.datetime(2021, 5, 1)
            # end_date = datetime.datetime(2023, 5, 19)
            for submission in reddit.subreddit(userText).hot(limit = 100):
            #     print(submission.title)
            #     print(submission.id)
            #     print(submission.author)
            #     print(submission.created_utc)
            #     print(submission.score)
            #     print(submission.upvote_ratio)
            #     print(submission.url)
                time = submission.created
                time1 = datetime.fromtimestamp(time)
                if start_date <= time1 <= end_date:
                    headlines.add(submission.title)
            head = list(headlines)
            data = {'Tweet' : head}
            file = pd.DataFrame(data)
            file.to_csv(f"{userText}.csv", index = False)
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
