import EmotionAnalysis as emotion
import SentimentAnalysis as sent
import TextClassification as text
import string
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


class model_input(BaseModel):
    file: UploadFile = File(...)
    target: string

# ----------------------------------------- text models endpoints --------------------------------
@app.post("/text_classification-model")
async def text_classification_endpoint(file: UploadFile = File(...), target: str = Form(...)):
    try:
        df = pd.read_csv(file.file)
        # Call the churn_prediction function
        fig = text.text_classification(df, target)
        # Convert the Plotly figure to JSON
        fig_json = iv.to_json(fig)
        return {"text_classification": fig_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"text_classification Error: {str(e)}")


@app.post("/emotion-analysis-model")
async def emotion_analysis_endpoint(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # Call the churn_prediction function
        fig = emotion.Emotion_analysis(df)
        # Convert the Plotly figure to JSON
        fig_json = iv.to_json(fig)
        return {"emotion_analysis": fig_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"emotion_analysis Error: {str(e)}")


@app.post("/sentiment-analysis-model")
async def sentiment_analysis_endpoint(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # Call the churn_prediction function
        fig = sent.sentiment_analysis(df)
        # Convert the Plotly figure to JSON
        fig_json = iv.to_json(fig)
        return {"sentiment_analysis": fig_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sentiment_analysis Error: {str(e)}")