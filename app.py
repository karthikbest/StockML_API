from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pandas as pd
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODELS_FOLDER = "models"
TIME_STEPS = 60  # Number of time steps for LSTM


# Function to fetch stock data from YFinance
def fetch_stock_data(ticker):
    """
    Fetch historical stock data for the given ticker from YFinance.
    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :return: NumPy array containing historical stock data and dates.
    """
    import time
    
    # Try multiple approaches to fetch data
    for attempt in range(3):
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Try different period options if one fails
            periods_to_try = ["3mo", "6mo", "1y"]
            hist_data = None
            
            for period in periods_to_try:
                try:
                    # Download data using download method as alternative
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=100)  # Get at least 100 days
                    
                    # Try using yf.download which is sometimes more reliable
                    hist_data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False,
                        auto_adjust=True,
                        prepost=False,
                        threads=False
                    )
                    
                    if not hist_data.empty:
                        break
                        
                except Exception as period_error:
                    print(f"Failed with period {period}: {str(period_error)}")
                    continue
            
            # If download failed, try the Ticker method
            if hist_data is None or hist_data.empty:
                hist_data = stock.history(period="3mo", auto_adjust=True, prepost=False)
            
            # Check if data was retrieved
            if hist_data.empty:
                if attempt < 2:
                    time.sleep(2)  # Wait before retry
                    continue
                raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
            
            # Sort by date to ensure chronological order
            hist_data = hist_data.sort_index()
            
            # Handle both single-level and multi-level column indices
            if isinstance(hist_data.columns, pd.MultiIndex):
                close_prices = hist_data['Close'][ticker].values.reshape(-1, 1) if 'Close' in hist_data else hist_data['Close'].values.reshape(-1, 1)
            else:
                close_prices = hist_data['Close'].values.reshape(-1, 1)
            
            # Get dates in string format
            sorted_dates = [date.strftime('%Y-%m-%d') for date in hist_data.index]
            
            # Ensure we have enough data
            if len(close_prices) < TIME_STEPS:
                raise ValueError(f"Insufficient historical data for ticker '{ticker}'. Need at least {TIME_STEPS} days.")
            
            # Debugging: Print data info
            print(f"YFinance data fetched: {len(close_prices)} days of data for {ticker}")
            
            return close_prices, sorted_dates
            
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise ValueError(f"Error fetching data for ticker '{ticker}' after 3 attempts: {str(e)}")
            else:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)  # Wait before retry


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
