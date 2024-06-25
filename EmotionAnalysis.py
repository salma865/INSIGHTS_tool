# import libraries
import re
import pandas as pd
import inflect
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import plotly.express as px
import text2emotion as te
import plotly
import json
from fastapi import FastAPI, UploadFile, File, Form


# convert number into words
def convert_number(text):
    p = inflect.engine()
    temp_str = text.split()
    new_string = []
    for word in temp_str:
        # if word is a digit, convert the digit to number
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        else:
            new_string.append(word)
    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str


def preprocess_text(text, lemmatize=True):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove white spaces
    text = " ".join(text.split())
    # convert numbers
    text = convert_number(text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize or stem the tokens
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    else:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def emotion_analysis(text):
    emotions = te.get_emotion(text)
    return emotions


def visualize_emotion_distribution(data):

    # Aggregate emotion counts
    emotion_counts = {
        'Happy': 0,
        'Sad': 0,
        'Angry': 0,
        'Fear': 0,
        'Surprise': 0
    }

    for emotions in data['emotions']:
        for emotion, score in emotions.items():
            emotion_counts[emotion] += score

    # Convert counts to DataFrame for plotting
    emotion_df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count'])

    # Plotting with plotly
    # Bar chart
    fig_bar = px.bar(emotion_df, x='Emotion', y='Count', title='Emotion Distribution (Bar Chart)')
    fig_bar.update_layout(xaxis_title='Emotion', yaxis_title='Count')

    # Display the charts
    return fig_bar


def Emotion_analysis(data):
    df = data.loc[:500]
    df['text'] = df['text'].apply(preprocess_text)
    df['emotions'] = df['text'].apply(emotion_analysis)
    fig = visualize_emotion_distribution(df)
    return fig


# Data = pd.read_csv("all_tweets.csv")
# fig = Emotion_analysis(Data)
# fig.show()


