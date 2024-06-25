import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from itertools import product
import preprocessing as prepro
import warnings
warnings.filterwarnings("ignore")


def check_stationary(data, target):
    adf_test = adfuller(data[target])
    if adf_test[1] > 0.05:
        data[target].diff().dropna()


def check_seasonality(data):
    acf_values = acf(data, fft=True, nlags=24)
    seasonal_lag = 12
    seasonal_peak = acf_values[seasonal_lag]

    decomposition = seasonal_decompose(data, model='additive', period=12)
    seasonal_variance = decomposition.seasonal.var()
    total_variance = data.var()
    seasonal_ratio = seasonal_variance / total_variance

    return seasonal_peak > 0.3 and seasonal_ratio > 0.1


def grid_search_arima(train, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    best_model = None

    for p, d, q in product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_score:
                best_score, best_cfg = aic, order
                best_model = model_fit
        except:
            continue
    return best_model, best_cfg


def visualize_arima(data, model):
    # Forecast future values
    forecast_steps = 30  # Example: forecasting 10 steps ahead
    forecast = model.forecast(steps=forecast_steps)
    last_date = data.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=pd.infer_freq(data.index))[1:]
    forecasted = go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecasted')
    layout = go.Layout(title='Predicted Revenue',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title='Revenue'))
    fig = go.Figure(data=forecasted, layout=layout)
    return fig


def evaluate_model_arima(data):
    p_values = range(0, 4)
    d_values = range(0, 2)
    q_values = range(0, 4)
    split_index = int(len(data) * 0.8)
    train, test = data[:split_index], data[split_index:]
    best_model, best_cfg = grid_search_arima(train, p_values, d_values, q_values)
    # Fit final model with best params on entire data
    final_model = ARIMA(data, order=best_cfg)
    final_fitted = final_model.fit()
    return final_fitted


def time_series_analysis(data, target):
    check_stationary(data, target)
    final_fitted = evaluate_model_arima(data[target])
    fig = visualize_arima(data[target], final_fitted)
    fig.show()
    return fig


# Data = pd.read_csv('women_clothing_ecommerce_sales.csv')
# numerical_stat, categorical_stat, Preprocessed_data = prepro.preprocessing(Data)
# time_series_analysis(Data, 'revenue')
