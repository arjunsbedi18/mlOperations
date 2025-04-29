import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    # try:
    #     # Set the MLFlow tracking URI
    #     # mlflow.set_tracking_uri('https://mlflow-test-run-275570243848.us-west2.run.app')
        
    #     # Load the model from the MLFlow model registry
    #     model = mlflow.sklearn.load_model("models:/metaflow-wine-model/latest")
    #     print("Model loaded successfully from MLFlow!")
    # except Exception as e:
    #     print(f"Error loading model: {e}")
    #     raise RuntimeError("Failed to load model from MLFlow")

@app.get("/")
async def root():
    return {"message": "Wine Classifier API"}

@app.post("/predict")
async def predict(wine_data: WineFeatures):
    try:
        # Ensure the input has the correct number of features (13 for wine dataset)
        # if len(wine_data.features) != 13:
        #     raise HTTPException(
        #         status_code=400, 
        #         detail=f"Expected 13 features for wine dataset, got {len(wine_data.features)}"
        #     )
        
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
# import mlflow
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# # Define the input data model
# class WineFeatures(BaseModel):
#     features: list[float]

# # Create FastAPI app
# app = FastAPI(title="Wine Classifier API")

# # Load the model from MLFlow at startup
# @app.on_event("startup")
# async def startup_event():
#     global model
#     try:
#         # Set the MLFlow tracking URI
#         mlflow.set_tracking_uri('https://mlflow-test-run-275570243848.us-west2.run.app')
        
#         # Load the model from the MLFlow model registry
#         model = mlflow.sklearn.load_model("models:/metaflow-wine-model/latest")
#         print("Model loaded successfully from MLFlow!")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         raise RuntimeError("Failed to load model from MLFlow")

# @app.get("/")
# async def root():
#     return {"message": "Wine Classifier API"}

# @app.post("/predict")
# async def predict(wine_data: WineFeatures):
#     try:
#         # Ensure the input has the correct number of features (13 for wine dataset)
#         if len(wine_data.features) != 13:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Expected 13 features for wine dataset, got {len(wine_data.features)}"
#             )
        
#         # Convert input to numpy array
#         features = np.array([wine_data.features])
        
#         # Make prediction
#         prediction = model.predict(features)[0]
        
#         # Return the prediction
#         return {
#             "prediction": int(prediction),
#             "prediction_label": f"Wine class {int(prediction)}"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000) 