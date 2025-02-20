import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
import streamlit as st
from keras.models import load_model

start = '2010-01-01'
end = '2025-01-01'

st.title("Stock Trend Prediction")

#Take user input
user_input = st.text_input('Enter the stock ticker:', 'AAPL')
df = yf.download(user_input, start, end)

#Describe data
st.subheader("Data from 2010 - 2025")
st.write(df.describe())

#Data Visualization
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Plot Moving Average of 100 days
st.subheader("Closing Price vs Time chart with 100 Moving Average")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

#Plot Moving Average of 200 days
st.subheader("Closing Price vs Time chart with 100 and 200 Moving Average")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
st.pyplot(fig)

#Split data into train and test
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

#scaling the train data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#load the model
model = load_model('keras_model.h5')

#Test data 
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

#prediction on test data and scale the test data
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final visualization 
st.subheader("Prediction vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)