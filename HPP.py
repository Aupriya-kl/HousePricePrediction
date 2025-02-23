from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import uvicorn

# Load the trained model
with open('house_price_model.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define a request model
class HouseFeatures(BaseModel):
    price: float
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    street: float
    city: float
    statezip: float
    country: float

# StandardScaler (must use same scaling as training data)
scaler = StandardScaler()

@app.post("/predict")
def predict_price(data: HouseFeatures):
    # Convert input to NumPy array and reshape for model prediction
    input_data = np.array([
        data.bedrooms, data.bathrooms, data.sqft_living, data.sqft_lot,
        data.floors, data.waterfront, data.view, data.condition,
        data.sqft_above, data.sqft_basement, data.yr_built,
        data.yr_renovated, data.street, data.city,
        data.statezip, data.country
    ]).reshape(1, -1)
    
    # Scale input features (assuming StandardScaler was used during training)
    input_data = scaler.fit_transform(input_data)  # Use transform if using pre-fitted scaler
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return {"predicted_price": prediction[0]}
if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)
