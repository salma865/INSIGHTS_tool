import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import stats


def types_splitting(data):
    nums = list(data.select_dtypes(['int64', 'float64']))
    cats = list(data.select_dtypes(['object']))
    return nums, cats


def drop_ids(data):
    for col in data:
        if "id" in col.lower():
            data.drop(columns=col, inplace=True)


def handling_duplicates(data):
    data.drop_duplicates(inplace=True)


def cats_encoding(data, cats):
    for col in cats:
        Encoder = LabelEncoder()
        data[col] = Encoder.fit_transform(data[col])


def handling_nulls(data, nums, cats):
    null_total = data.isnull().sum()
    per = null_total / len(data)
    col_to_drop = per[per > 0.5].index
    nums = [x for x in nums if x not in col_to_drop]
    cats = [x for x in cats if x not in col_to_drop]
    data.drop(columns=col_to_drop, inplace=True)
    for col in nums:
        data[col] = data[col].fillna(data[col].median())
    for col in cats:
        data[col] = data[col].fillna(data[col].mode()[0])


def normalize(data, nums):
    scaler = MinMaxScaler(feature_range=(0, 1))
    for col in nums:
        norm = scaler.fit_transform(data[col])


def outlire(data):
    Q1 = data.quantile(q=.25)
    Q3 = data.quantile(q=.75)
    IQR = data.apply(stats.iqr)

    data_clean = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def wrong_format(data):
    datetime_columns = data.select_dtypes(include=['datetime']).columns.tolist()

    for col in datetime_columns:
        data[col + '_time'] = data[col].dt.time
        data[col + '_hour'] = data[col + '_time'].apply(lambda x: x.hour)
        data[col + '_minute'] = data[col + '_time'].apply(lambda x: x.minute)
        data[col + '_second'] = data[col + '_time'].apply(lambda x: x.second)
        data[col + '_day'] = data[col].dt.day
        data[col + '_month'] = data[col].dt.month
        data[col + '_year'] = data[col].dt.year


def heatmap(data):
    corr = data.corr()
    fig = px.imshow(corr, text_auto=True)
    return fig


def preprocessing(data):
    drop_ids(data)
    numerical, categorical = types_splitting(data)
    handling_nulls(data, numerical, categorical)
    handling_duplicates(data)
    numerical_statistics = num_stat(data, numerical)
    categorical_statistics = cat_stat(data, categorical)
    cats_encoding(data, categorical)
    fig=heatmap(data)
    return numerical_statistics, categorical_statistics,data,fig


def cat_stat(data, cats):
    stat = {}
    for col in cats:
        col_stat = data[col].value_counts(normalize=True)*100
        col_stat = col_stat.round(2).astype(str)+'%'
        stat[col] = col_stat
    return stat


def num_stat(data, nums):
    stat = {}
    for col in nums:
        col_stat = {}
        col_stat['Average'] = data[col].mean()
        col_stat['Min'] = data[col].min()
        col_stat['Max'] = data[col].max()
        stat[col] = col_stat
    return stat
