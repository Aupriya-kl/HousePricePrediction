# HousePricePrediction

## Overview
This project provides a FastAPI-based machine learning API for predicting house prices based on various features such as the number of bedrooms, bathrooms, square footage, location, and more. The model used for prediction is a trained `LinearRegression` model.

## Features
- Accepts house feature inputs via a RESTful API endpoint.
- Scales input features before making predictions.
- Returns the predicted house price as JSON output.
- Uses FastAPI for fast and efficient API deployment.

## Requirements
To run this API, install the required dependencies using:

```bash
pip install fastapi uvicorn numpy pandas scikit-learn pydantic
```

## Running the API
Start the FastAPI server using Uvicorn:

```bash
uvicorn filename:app --reload
```

Replace `filename` with the actual Python script name.

## API Endpoint
### `POST /predict`
#### Request Body (JSON):
```json
{
    "price": 0.0,
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1500,
    "sqft_lot": 5000,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1200,
    "sqft_basement": 300,
    "yr_built": 2000,
    "yr_renovated": 0,
    "street": 5,
    "city": 10,
    "statezip": 98001,
    "country": 1
}
```

#### Response:
```json
{
    "predicted_price": 450000.0
}
```

## Model Training
The model is trained using `LinearRegression` from `scikit-learn`. Data preprocessing includes encoding categorical variables and standardizing numerical features using `StandardScaler`.

## Notes
- Ensure the model is saved as `house_price_model.pkl` in the same directory.
- The input features should be standardized using the same scaler used during training.
- Modify the API as needed to integrate with a frontend or database.

## Future Enhancements
- Deploy as a cloud-based service.
- Improve model accuracy with advanced regression techniques.
- Add more detailed logging and error handling.

## License
This project is open-source and free to use for educational and research purposes.
