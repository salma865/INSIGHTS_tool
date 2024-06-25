# import libraries
import re
import pandas as pd
import inflect
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px


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


def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def sentiment_results(data):
    sentiment_counts = data['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment_category', 'count']

    # Visualize the sentiment distribution using Plotly
    fig = px.bar(sentiment_counts,
                 x='sentiment_category', y='count',
                 labels={'sentiment_category': 'Sentiment', 'count': 'Number of Reviews'},
                 title='Sentiment Analysis of Reviews')
    return fig


def sentiment_analysis(data):
    data['text'] = data['text'].apply(preprocess_text)
    sid = SentimentIntensityAnalyzer()
    data['sentiment'] = data['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    data['sentiment_category'] = data['sentiment'].apply(categorize_sentiment)
    fig = sentiment_results(data)
    return fig


# Data = pd.read_csv("all_tweets.csv")
# fig = sentiment_analysis(Data)
# fig.show()



