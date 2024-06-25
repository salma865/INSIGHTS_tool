import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import inflect
import string
import re


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


def processing(df, text, label):
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(label)
    vectorizer = CountVectorizer(max_features=2000)
    X = vectorizer.fit_transform(text).toarray()
    return X, encoded


def train_test_splt(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, labels)
    return X_train, X_test, y_train, y_test


def svm_classifier(X_train, X_test, y_train, y_test):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def create_barchart(data, x, y, dist=None, mode=None, color=None):
    fig = px.bar(data_frame=data, x=x, y=y, color=dist, barmode=mode, color_discrete_sequence=color)
    return fig


def plot(pred, labels):
    datalabels = labels.unique()
    li = []
    li2 = []
    for i in range(len(datalabels)):
        new = []
        for j in range(len(pred)):
            if i == pred[j]:
                new.append(i)
        li2.append(new.count(i))
        li.append(new)
    labels_for_plot = datalabels
    counts_for_plot = li2

    data = pd.DataFrame({
        'Category': labels_for_plot,
        'Count': counts_for_plot
    })

    x = 'Category'
    y = 'Count'
    dist = None
    mode = None
    color = ['blue', 'green']

    fig = create_barchart(data, x, y, dist, mode, color)

    fig.update_layout(
        title='text classification',
        xaxis_title='Category',
        yaxis_title='Count'
    )
    return fig


def text_classification(data, target):
    data['text'] = data['text'].apply(preprocess_text)
    X, labels = processing(data, data['text'], data[target])
    X_train, X_test, y_train, y_test = train_test_splt(X, labels)
    acc, pred = svm_classifier(X_train, X_test, y_train, y_test)
    fig = plot(pred, data[target])
    return fig


# df = pd.read_csv('TextClassificationData.csv')
# fig = text_classification(df, 'label')
# fig.show()

