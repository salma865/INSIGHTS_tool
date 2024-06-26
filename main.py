from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from pydantic import BaseModel
import uvicorn
import visualization as iv
import json
import pandas as pd
import preprocessing as pre
import numpy as np

app = FastAPI()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # np series
        if isinstance(obj, pd.Series):
            return obj.to_list()
        return super(NpEncoder, self).default(obj)


# -----------------------------endpoint_histogram-------------------
@app.post("/create-histogram")
async def create_histogram_endpoint(data: dict = Form(...), x: str = Form(...), dist: str = Form(None),
                                    color: str = Form(None), bins: int = Form(50), func: str = Form('count'),
                                    norm: str = Form('')):
    fig = iv.create_histogram(data, x, dist, color, bins, func, norm)
    fig_json = iv.to_json(fig)
    return {"histogram": fig_json}


# ----------------------------endpoint_linechart-----------------------
@app.post("/create-linechart")
async def create_linechart_endpoint(data: dict = Form(...), x: str = Form(...), y: str = Form(...),
                                    dist: str = Form(None), shape: str = Form('linear'), color: str = Form(None)):
    fig = iv.create_linechart(data, x, y, dist, shape, color)
    fig_json = iv.to_json(fig)
    return {"linechart": fig_json}


# ----------------------------endpoint_barchart-----------------------
@app.post("/create-barchart")
async def create_barchart_endpoint(data: dict = Form(...), x: str = Form(...), y: str = Form(...),
                                   dist: str = Form(None), mode: str = Form(None), color: str = Form(None)):
    fig = iv.create_barchart(data, x, y, dist, mode, color)
    fig_json = iv.to_json(fig)
    return {"barchart": fig_json}


# ----------------------------endpoint_boxplot-------------------------
@app.post("/create-boxplot")
async def create_boxplot_endpoint(data: dict = Form(...), y: str = Form(...), dist: str = Form(None),
                                  points: str = Form(None), color: str = Form(None)):
    fig = iv.create_boxplot(data, y, dist, points, color)
    fig_json = iv.to_json(fig)
    return {"boxplot": fig_json}


# ----------------------------endpoint_pie_chart-------------------------
@app.post("/create-pie-chart")
async def create_pie_chart_endpoint(data: dict = Form(...), values: str = Form(...), names: str = Form(...),
                                    color: str = Form(None)):
    fig = iv.create_pie_chart(data, values, names, color)
    fig_json = iv.to_json(fig)
    return {"piechart": fig_json}


# ----------------------------endpoint_scatterplot-------------------------
@app.post("/create-scatter-plot")
async def create_scatter_plot_endpoint(data: dict = Form(...), x: str = Form(...), y: str = Form(...),
                                       dist: str = Form(None), size: str = Form(None),
                                       symbol: str = Form(None), color: str = Form(None)):
    fig = iv.create_scatter_plot(data, x, y, dist, size, symbol, color)
    fig_json = iv.to_json(fig)
    return {"scatterplot": fig_json}


# ---------------------------update_label-----------------------------------
@app.post("/update-labels")
async def update_labels_endpoint(fig_json: str = Form(...), x_label: str = Form(...), y_label: str = Form(...)):
    fig = json.loads(fig_json)
    updated_fig = iv.update_labels(fig, x_label, y_label)
    updated_fig_json = iv.to_json(updated_fig)
    return {"plot": updated_fig_json}


# ---------------------------update_title-----------------------------------
@app.post("/update-title")
async def update_title_endpoint(fig_json: str = Form(...), title: str = Form(...)):
    fig = json.loads(fig_json)
    updated_fig = iv.update_title(fig, title)
    updated_fig_json = iv.to_json(updated_fig)
    return {"plot": updated_fig_json}


# --------------------------------------------preprocessing_encoding-----------------
@app.post("/preprocessing")
async def preprocessing_endpoint(file: UploadFile = File(...), check: str = Form(...)):
    try:
        df = pd.read_csv(file.file)
        nums, cats = pre.types_splitting(df)
        if check == "numeric":
            # Perform preprocessing
            num_stats, cat_stats, preprocessed_data = pre.preprocessing(df)
            # Return the preprocessing results
            return {
                "numerical_statistics": json.loads(json.dumps(num_stats, cls=NpEncoder)),
                "categorical_statistics": json.loads(json.dumps(cat_stats, cls=NpEncoder)),
                "preprocessed_data": preprocessed_data.to_dict(orient="records"),
                "numerical_columns": nums,
                "categorical_columns": cats
            }
        else:
            return {
                "preprocessed_data": df.to_dict(orient="records"),
                "numerical_columns": nums,
                "categorical_columns": cats
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"time series Error: {str(e)}")

class InputValues(BaseModel):
    first: str = Form(...)
    second: str = Form(...)
    numerical: List[str]
    categorical: List[str]


class OutputValues(BaseModel):
    results: List[str]


# ---------------------------filter charts-----------------------------------
@app.post("/filter-charts")
async def filter_charts_endpoint(inputvalues: InputValues):
    vaild_charts = iv.filter_charts(
        inputvalues.first,
        inputvalues.second,
        inputvalues.numerical,
        inputvalues.categorical
    )
    return {"valid_charts": OutputValues(results=vaild_charts)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)