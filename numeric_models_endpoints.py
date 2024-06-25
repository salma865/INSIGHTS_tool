import string
import pandas as pd
import ChurnPrediction as churn
import CustomerSegmentation as customer
import TimeSeriesAnalysis as time
import visualization as iv
import preprocessing as preproc
from fastapi import FastAPI, HTTPException
import pandas as pd
import uvicorn
import json
import io
app = FastAPI()
from pydantic import BaseModel
from typing import Dict, List, Union
from fastapi import FastAPI, UploadFile, File, Form


# ------------------------------------------------ numeric models endpoints ------------------------------------------
@app.post("/churn-prediction-model")
async def churn_prediction_endpoint(file: UploadFile = File(...), target: str = Form(...)):
    try:
        df = pd.read_csv(file.file)
        # Call the churn_prediction function
        fig = churn.churn_prediction(df, target)
        # Convert the Plotly figure to JSON
        fig_json = iv.to_json(fig)
        return {"churn_prediction": fig_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Churn Error: {str(e)}")


@app.post("/time-series-model")
async def time_series_endpoint(file: UploadFile = File(...), target: str = Form(...)):
    try:
        df = pd.read_csv(file.file)
        # Call the churn_prediction function
        fig = time.time_series_analysis(df, target)
        # Convert the Plotly figure to JSON
        fig_json = iv.to_json(fig)
        return {"time_series": fig_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"time series Error: {str(e)}")


@app.post("/customer-segmentation-model")
async def customer_segmentation_endpoint(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # Call the churn_prediction function
        fig = customer.customer_segmentation(df)
        # Convert the Plotly figure to JSON
        fig_json = iv.to_json(fig)
        return {"customer_segmentation": fig_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"customer segmentation Error: {str(e)}")
