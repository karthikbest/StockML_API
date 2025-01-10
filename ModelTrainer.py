import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import requests
from bs4 import BeautifulSoup


# Step 1: Fetch the S&P 500 Tickers from Wikipedia
def get_sp500_tickers():
    """
    Fetches the list of S&P 500 company tickers from Wikipedia.
    :return: List of stock tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch S&P 500 tickers from Wikipedia.")

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []

    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text.strip()
        tickers.append(ticker)

    return tickers


# Step 2: Fetch Historical Stock Data
def fetch_stock_data(ticker):
    """
    Fetch historical stock data for a given ticker symbol.
    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :return: DataFrame containing historical stock data.
    """
    data = yf.download(ticker, period="5y", interval="1d")  # 5 years of daily data
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    return data[["Close"]]


# Step 3: Preprocess the Data
def preprocess_data(data, time_steps=60):
    """
    Prepares the data for LSTM training.
    :param data: DataFrame containing the "Close" price.
    :param time_steps: Number of previous time steps to use for predictions.
    :return: Tuple (X_train, y_train), scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(time_steps, len(scaled_data)):
        X_train.append(scaled_data[i - time_steps:i, 0])
        y_train.append(scaled_data[i, 0])
    return np.array(X_train), np.array(y_train), scaler


# Step 4: Build the LSTM Model
def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model.
    :param input_shape: Shape of the input data (time_steps, features).
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Step 5: Train and Save the Model
def train_and_save_model(ticker, models_folder, time_steps=60, epochs=10, batch_size=32):
    """
    Fetches data, trains an LSTM model, and saves the trained model.
    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :param models_folder: Directory to save the models.
    :param time_steps: Number of time steps for LSTM input.
    :param epochs: Number of epochs to train the model.
    :param batch_size: Batch size for training.
    """
    try:
        # Fetch historical data
        print(f"Fetching data for {ticker}...")
        data = fetch_stock_data(ticker)

        # Preprocess the data
        X_train, y_train, scaler = preprocess_data(data.values, time_steps)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the LSTM model
        print(f"Training model for {ticker}...")
        model = build_lstm_model((X_train.shape[1], 1))

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Save the trained model
        os.makedirs(models_folder, exist_ok=True)
        model.save(os.path.join(models_folder, f"{ticker}_lstm_model.h5"))
        print(f"Model for {ticker} saved to {models_folder}/{ticker}_lstm_model.h5.")
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")


# Main Script
if __name__ == "__main__":
    # Fetch all S&P 500 tickers
    tickers = get_sp500_tickers()
    models_folder = "models"

    # Train models for each ticker
    for ticker in tickers:
        train_and_save_model(ticker, models_folder)
