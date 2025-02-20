# Stock Trend Prediction Using Deep Learning

This project is a **Stock Trend Prediction Application** built with **Streamlit, Keras, and Yahoo Finance API**. It enables users to visualize stock price trends and make future predictions using a pre-trained deep learning model.

## Introduction

Stock price prediction is a challenging task in financial markets. This project uses **LSTM-based deep learning models** to analyze stock price movements and provide trend predictions based on historical data.

## Features

- **User Input:** Users can enter any stock ticker symbol to analyze trends.
- **Data Fetching:** Fetches real-time stock data from Yahoo Finance.
- **Data Visualization:** Provides moving average plots and historical data analysis.
- **Deep Learning Model:** Predicts future stock prices using an LSTM model trained on historical stock data.
- **Interactive UI:** Built with Streamlit for easy and interactive user experience.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Riddhikshah21/Stock-price-prediction-using-LSTM-web-app.git
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the pre-trained model:**
   Ensure the `keras_model.h5` file is present in the project directory.

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
2. **Enter a stock ticker symbol** (e.g., AAPL, MSFT) in the input box.
3. **View visualizations:** The app will display stock trends with moving averages and predictions.

## Dataset

The application retrieves stock market data from **Yahoo Finance** using the `yfinance` Python library. The model is trained using historical stock prices from **2010 to 2025**.

## Model Details

- Uses a **Long Short-Term Memory (LSTM)** model trained on scaled stock prices.
- Implements **MinMaxScaler** to normalize data before training.
- Trained on stock price history with a **100-day moving window**.

## Results

The model predicts future stock trends based on historical data. The final visualization includes:

- **Original vs Predicted Stock Prices**
- **Moving Average Trends** (100-day & 200-day)
