from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import preprocessing as prepro


def split_train_test(df, label_column):
    x = df.drop(label_column, axis=1)
    y = df[label_column]
    count_y_0 = list(y).count(0)
    count_y_1 = list(y).count(1)
    if abs(count_y_0 - count_y_1) > 100:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(x, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_logistic_regression(df, target, penalty='l2', C=1.0, max_iter=100):
    X_train, X_test, y_train, y_test = split_train_test(df, target)
    model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy


def train_svm_regression(df, target, kernel='linear', probability=True):
    X_train, X_test, y_train, y_test = split_train_test(df, target)
    svm = SVC(kernel=kernel, probability=probability)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    return y_pred_svm, accuracy_svm


def create_barchart(data, x, y, dist=None, mode=None, color=None):
    fig = px.bar(data_frame=data, x=x, y=y, color=dist, barmode=mode, color_discrete_sequence=color)
    return fig


def churn_prediction(df, target):
    df.drop(columns='id', inplace=True)
    p1, acc1 = train_logistic_regression(df, target)
    p2, acc2 = train_svm_regression(df, target)
    if acc1 > acc2:
        count_zeros = list(p1).count(0)
        count_ones = list(p1).count(1)
        data = pd.DataFrame({
            'Category': ['Will Churn', 'Will Not Churn'],
            'Count': [count_zeros, count_ones]
        })
        x = 'Category'
        y = 'Count'
        dist = None
        mode = None
        color = ['blue', 'green']
        fig = create_barchart(data, x, y, dist, mode, color)
        fig.update_layout(title='Not Included Customers Vs Included Customers', xaxis_title='Category',
                          yaxis_title='Count')
    else:
        count_zeros = list(p2).count(0)
        count_ones = list(p2).count(1)
        data = pd.DataFrame({
            'Category': ['Will Churn', 'Will Not Churn'],
            'Count': [count_zeros, count_ones]
        })
        x = 'Category'
        y = 'Count'
        dist = None
        mode = None
        color = ['blue', 'green']
        fig = create_barchart(data, x, y, dist, mode, color)
        fig.update_layout(
            title='Customer Churn',
            xaxis_title='Category',
            yaxis_title='Count'
        )
    return fig


#Data = pd.read_csv('ChurnPrediction.csv')
#numerical_stat, categorical_stat, Preprocessed_data = prepro.preprocessing(Data)
#fig = churn_prediction(Data, 'Churn')
#fig.show()
