from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pickle
import pandas as pd
from typing import List, Dict, Any
import os
import json

app = FastAPI(title='Reder API', version='1.0.0')

# define request body
class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        example= [{
            'payment_risk_score': 25.00,
            'total_interactions': 12,
            'TimeSpent(minutes)': 28,
            'NPS': 6,
            'engagement_intensity': 392,
            'engagement_ratio': 1.866667,
            'nps_category_Detractor': 1.0,
            'nps_category_Passive': 0.0,
            'last_interaction_month': 12,
            'nps_category_Promoter': 0.0,
            'customer_segment': 1.0,
            'last_interaction_dayofweek': 4,
            'Gender_Male': 0.0,
            'emails_opened': 10,
            'PageViews': 14,
            'page_diversity': 0.311111,
            'Segment_Segment C': 1.0,
            'search_count': 13
        }]
    )

def load_model():
    model_path = os.path.join('model', 'model.pkl')
    feature_path = os.path.join('model', 'data.json')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(feature_path, 'r') as file:
        features = json.load(file)

    return model, features

@app.get('/testing')
def read_root():
    return {"Hello": "World"}

@app.post('/predict')
def predict(req: PredictionRequest):
    # convert incoming request to dataframe
    df = pd.DataFrame(req.records)

    # load model
    model, features = load_model()

    # matching features at train time with features at inference
    df = df.reindex(columns= features, fill_value= 0)

    # make prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    print(prediction)
    print(type(prediction))

    print(prediction_proba)
    print(type(prediction_proba))

    # return prediction and predictiion_proba as response
    return {'prediction': int(prediction[0]), 'prediction_probability': float(prediction_proba[:, 1][0])}

if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=True)
