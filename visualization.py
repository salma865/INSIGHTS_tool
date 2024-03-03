import plotly
import plotly.express as px
import json


# ------------------------------ Histogram ------------------------------
def create_histogram(data, x, dist=None, color=None, bins=50, func='count', norm=''):
    fig = px.histogram(data_frame=data, x=x, color=dist, color_discrete_sequence=color, nbins=bins, histfunc=func, histnorm=norm)
    return fig


# ------------------------------ Line chart ------------------------------
def create_linechart(data, x, y, dist=None, shape='linear', color=None):
    fig = px.line(data_frame=data, x=x, y=y, color=dist, markers=True, line_shape=shape, color_discrete_sequence=color)
    return fig


# ------------------------------ Bar chart ------------------------------
def create_barchart(data, x, y, dist=None, mode=None, color=None):
    fig = px.bar(data_frame=data, x=x, y=y, color=dist, barmode=mode, color_discrete_sequence=color)
    return fig


# ------------------------------ Box Plot ------------------------------
def create_boxplot(data, y, dist=None, points=None, color=None):
    fig = px.box(data_frame=data, y=y, x=dist, points=points, color_discrete_sequence=color)
    return fig


# ------------------------------ Pie Chart ------------------------------
def create_pie_chart(data, values, names, color=None):
    fig = px.pie(data_frame=data, values=values, names=names, color_discrete_sequence=color)
    return fig


# ------------------------------ Scatter Plot ------------------------------
def create_scatter_plot(data, x, y, dist=None, size=None, symbol=None, color=None):
    fig = px.scatter(data_frame=data, x=x, y=y, color=dist, size=size, symbol=symbol, color_discrete_sequence=color)
    return fig


# ------------------------------ Labels ------------------------------
def update_labels(fig, x_label, y_label):
    fig.update_layout(xaxis_title_text=x_label, yaxis_title_text=y_label)
    return fig


# ------------------------------ Title ------------------------------
def update_title(fig, title):
    fig.update_layout(title=title)
    return fig


# ------------------------------ Json ------------------------------
def to_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
