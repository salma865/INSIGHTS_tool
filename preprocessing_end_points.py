from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import preprocessing as pre
import visualization as iv


app = FastAPI()
# --------------------------------------------preprocessing_encoding------------------


@app.post("/preprocessing")
async def preprocessing_endpoint(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Perform preprocessing
    num_stats, cat_stats, preprocessed_data, fig = pre.preprocessing(df)
    fig_json = iv.to_json(fig)
    # Return the preprocessing results
    return {
        "numerical_statistics": num_stats,
        "categorical_statistics": cat_stats,
        "preprocessed_data": preprocessed_data.to_dict(orient="records"),
        "heatmap": fig_json}


# ------------------------------------------types_splitting_encoding------------------------
@app.post("/types_splitting")
async def types_splitting_endpoint(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    # Apply types_splitting function
    nums, cats = pre.types_splitting(df)
    # Return the result
    return {"numerical_columns": nums, "categorical_columns": cats}
