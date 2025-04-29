import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine

# Define the input data model
class WineFeatures(BaseModel):
    features: list[float]

# Create FastAPI app
app = FastAPI(title="Wine Classifier API")

# Load the model from MLFlow at startup
@app.on_event("startup")
async def startup_event():
    global model
    wine = load_wine()
    y = wine.target
    X = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X, y)

@app.get("/")
async def root():
    return {"message": "Wine Classifier API"}

@app.post("/predict")
async def predict(wine_data: WineFeatures):
    try:
        
        # Convert input to numpy array
        features = np.array([wine_data.features])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Return the prediction
        return {
            "prediction": int(prediction),
            "prediction_label": f"Wine class {int(prediction)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
