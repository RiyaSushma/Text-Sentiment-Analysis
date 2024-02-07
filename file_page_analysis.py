import streamlit as st
import pandas as pd
import numpy as np
import nltk
import plotly.express as px
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import string

import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

# Download the NLTK tokenizer and stop words if you haven't already
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model
model = keras.models.load_model('sentiment_analysis_trail_model.h5')
#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
   ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

#Emoji patterns
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)


#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_mentions(text):
    # remove mentions (words that start with '@')
    return re.sub(r'@\w+', '', text)


def clean_tweets(tweet):
    # Remove non-ASCII characters
    tweet = remove_mentions(tweet)
    tweet = re.sub('[^\x00-\x7F]+', ' ', tweet)
    
    # Remove URLs
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = emoji_pattern.sub(r'', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize tweet
    word_tokens = word_tokenize(tweet.lower())
    
    # Lemmatize words and remove stop words, emoticons, and punctuation
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words and w not in string.punctuation:
            w = re.sub(r'@\w+', '', w)
            filtered_tweet.append(w)

    return ' '.join(filtered_tweet)

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

# create a dictionary to store the mappings between the original words and their cleaned versions
word_dict = {}

def get_word(word):
    if word not in word_dict:
        pos = pos_tag([word])[0][1]
        if pos.startswith('J'):
            clean_word = lemmatizer(word, get_simple_pos(pos))[0]
            word_dict[word] = clean_word.lower()
        else:
            word_dict[word] = ""
    return word_dict[word]

def get_tag(text):
    output_words = [get_word(word) for word in text]
    return output_words


# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenize the text
#     words = nltk.word_tokenize(text)
#     # Remove stop words
#     words = [word for word in words if word not in stopwords.words('english')]
#     # Join the words back into a string
#     text = " ".join(words)
#     return text

# Create a function to predict the sentiment
def predict_sentiment(text):
    # Preprocess the text
    text = clean_tweets(text)
    
    # Create a tokenizer instance
    tokenizer = Tokenizer()
    # Fit the tokenizer on the preprocessed text
    tokenizer.fit_on_texts([text])
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    sequence_padded = pad_sequences(sequence, maxlen=162)
    # Predict the sentiment
    prediction = model.predict(sequence_padded)
    pred_class = np.argmax(prediction, axis = 1)[0]
    # Get the label and the percentages for each sentiment
    # label = np.argmax(prediction)
    # percent_positive = prediction[0]
    # percent_negative = prediction[1]
    # percent_neutral = prediction[2]
    # return percent_positive, percent_negative, percent_neutral
    return pred_class

# Create a function to read the input file
# Create a function to read the input file
def read_input_file(file):
    # Read the file into a dataframe
    df = pd.read_csv(file, encoding='ISO-8859-1')
    # Check if the dataframe has the expected column names
    expected_columns = {'Tweet'}
    if not set(expected_columns).issubset(set(df.columns)):
        # If the column names don't match, raise an error
        raise ValueError(f'The input file must have the following columns: {expected_columns}')
    # Return the text column as a list
    return df['Tweet'].tolist()


# Create the Streamlit app
# Create the Streamlit app
def file_page_A(file):
    sentences = read_input_file(file)
    # Create a list to store the results
    results = []
    # Loop through the sentences
    count_pos = 0
    count_neg = 0
    count_neut = 0
    # Loop through the sentences
    for sentence in sentences:
            label = predict_sentiment(sentence)

            # Predict the sentiment
            # percent_positive, percent_negative, percent_neutral = predict_sentiment(sentence)
            # Format the output
            if label == 0:
                label_text = 'Positive'
                count_pos += 1
            elif label == 1:
                label_text = 'Negative'
                count_neg += 1
            else:
                label_text = 'Neutral'
                count_neut += 1
            # results.append({'Sentence': sentence, 'Sentiment': label_text, 'Negative': percent_negative, 'Neutral': percent_neutral, 'Positive': percent_positive})
            results.append({'Sentence': sentence, 'Label': label_text})
        #     results.append({'Sentence' : sentence, 'Negative': percent_negative, 'Neutral': percent_neutral, 'Positive': percent_positive})
        # # Convert the results to a dataframe
    df_results = pd.DataFrame(results)
    st.write(df_results)
    # df_results = pd.DataFrame(results)
    # # Show the results in a table
    # st.write(df_results)

    # Show the results in a bar chart
    categ = ['Postive', 'Negative', 'Neutral']
    chart_data = [count_pos, count_neg, count_neut]
    explode = (0.1, 0.2, 0.1)
    colors = ("orange", "cyan", "blue")
    wp = {'linewidth' : 1, 'edgecolor' : 'green'}
    fig, ax = plt.subplots(figsize = (10, 6))
    def func(pct, chart_data):
        absolute = int(pct / 100.*np.sum(chart_data))
        return "{:1f}%\n({:d} g)".format(pct, absolute)
    wedges, text, autotexts = ax.pie(
        chart_data,
        autopct=lambda pct: func(pct, 
        chart_data),
        explode = explode,
        labels = categ,
        shadow = True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        textprops=dict(color = "black")
    )
    ax.legend(wedges, categ,
      title ="Sentiment",
      loc ="center left",
      bbox_to_anchor =(1, 0, 1, 2))
        
    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Sentiment Analysis")

 
    st.pyplot(fig)
        
    # Show the overall sentiment percentages
    total = count_pos + count_neg + count_neut
    percent_positive = (count_pos * 100)/total
    percent_negative = (count_neg * 100)/total
    percent_neutral = (count_neut * 100)/total
    st.write(f'Overall Sentiment: Negative={percent_negative:.2f}%, Neutral={percent_neutral:.2f}%, Positive={percent_positive:.2f}%')

if __name__ == '__file_page_A__':
    file_page_A()
