from fastapi import FastAPI, UploadFile, File, Form,HTTPException
import pandas as pd
import preprocessing as pre
import visualization as iv


app = FastAPI()
# --------------------------------------------preprocessing_encoding------------------


@app.post("/preprocessing")
async def preprocessing_endpoint(file: UploadFile = File(...), check: str = Form(...)):

    try:
        if check == "numeric":
            df = pd.read_csv(file.file)
            # Perform preprocessing
            num_stats, cat_stats, preprocessed_data = pre.preprocessing(df)
            # Return the preprocessing results
            return {
                "numerical_statistics": num_stats,
                "categorical_statistics": cat_stats,
                "preprocessed_data": preprocessed_data.to_dict(orient="records")
            }
        else:
            df = pd.read_csv(file.file)
            return {"df": df}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"time series Error: {str(e)}")


# ------------------------------------------types_splitting_encoding------------------------
@app.post("/types_splitting")
async def types_splitting_endpoint(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    # Apply types_splitting function
    nums, cats = pre.types_splitting(df)
    # Return the result
    return {"numerical_columns": nums, "categorical_columns": cats}
