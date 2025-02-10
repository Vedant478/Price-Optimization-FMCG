import json
import joblib
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from azureml.core.model import Model

# Load the model when the container starts
def init():
    global model
    model_path = Model.get_model_path('best_xgb_2')
    model = joblib.load(model_path)

# Run inference when an HTTP request is received
def run(raw_data):
    try:
        # Convert input JSON to pandas DataFrame
        data = pd.DataFrame(json.loads(raw_data))
        
        # Make predictions
        predictions = model.predict(data)
        
        # Return predictions as JSON
        return json.dumps(predictions.tolist())
    
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
