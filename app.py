from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODELS_FOLDER = "models"
TIME_STEPS = 60  # Number of time steps for LSTM
ALPHA_VANTAGE_API_KEY = "L7T0XECYN3RJ2T7D"  # Replace with your actual API key


# Function to fetch stock data from Alpha Vantage
def fetch_stock_data(ticker):
    """
    Fetch historical stock data for the given ticker from Alpha Vantage (Free Tier).
    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :return: NumPy array containing historical stock data.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",  # Use the free version
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"  # "full" returns all data, "compact" returns last 100 days
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Debugging: Print API response to check for issues
    print("Alpha Vantage Response:", data)

    # Check for errors
    if "Time Series (Daily)" not in data:
        error_message = data.get("Note", "Invalid API response. Check API key and ticker symbol.")
        raise ValueError(f"No data found for ticker '{ticker}'. {error_message}")

    stock_data = data["Time Series (Daily)"]
    sorted_dates = sorted(stock_data.keys(), reverse=False)  # Ensure chronological order
    close_prices = [[float(stock_data[date]["4. close"])] for date in sorted_dates]

    return np.array(close_prices), sorted_dates


# Function to preprocess data
def preprocess_data(data, time_steps):
    """
    Prepare the stock data for LSTM prediction.
    :param data: NumPy array with the "Close" column.
    :param time_steps: Number of time steps for LSTM input.
    :return: Tuple of (scaled last sequence, scaler).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < time_steps:
        raise ValueError("Not enough data to create sequences for LSTM input.")

    last_sequence = scaled_data[-time_steps:]
    return np.reshape(last_sequence, (1, time_steps, 1)), scaler


@app.route('/')
def home():
    return "Welcome to Stock Price Prediction API. Please refer to API documentation for usage!"


@app.route('/predict', methods=['GET'])
def predict():
    """
    Predict stock prices for the next 5 days using a pre-trained model.
    Query Parameters:
    - company: Stock ticker symbol (e.g., "AAPL").
    """
    ticker = request.args.get("company")
    if not ticker:
        return jsonify({"error": "No company name provided."}), 400

    try:
        # Construct the model path
        model_path = os.path.join(MODELS_FOLDER, f"{ticker}_lstm_model.h5")

        # Check if the model exists
        if not os.path.exists(model_path):
            return jsonify({"error": f"No pre-trained model found for ticker '{ticker}'."}), 404

        # Load the pre-trained model
        model = load_model(model_path)

        # Fetch historical stock data
        data, dates = fetch_stock_data(ticker)

        # Preprocess the data for prediction
        last_sequence, scaler = preprocess_data(data, TIME_STEPS)

        # Predict the next 5 days
        predictions = []
        for _ in range(5):
            pred_price = model.predict(last_sequence)[0, 0]
            predictions.append(pred_price)
            last_sequence = np.append(last_sequence[:, 1:, :], [[[pred_price]]], axis=1)

        # Rescale predictions back to original range
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

        # Extract the last closing price
        last_closing_price = float(data[-1, 0])
        last_closing_date = dates[-1]

        # Generate future dates for the predicted prices
        prediction_dates = [
            (datetime.strptime(last_closing_date, '%Y-%m-%d') + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in
            range(5)]

        # Generate a recommendation
        recommendation = "Buy" if predictions[-1] > last_closing_price else "Sell"

        # Return the response
        return jsonify({
            "CompanyName": ticker.upper(),
            "LastClosingPrice": {
                "Date": last_closing_date,
                "Price": round(last_closing_price, 2)
            },
            "PredictedPrices": [
                {"Date": date, "Price": round(price, 2)} for date, price in zip(prediction_dates, predictions)
            ],
            "Recommendation": recommendation
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
