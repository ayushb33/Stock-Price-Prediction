import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow import keras
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

model = keras.models.load_model('C:\minor project-stock prediction\Stock Prediction Model.keras')

st.header('Stock Price Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')

start = '2014-01-01'
end = '2024-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):int(len(data))])

scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

data_test_scale = scaler.fit_transform(data_test)

# Moving Averages for 50 days
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()

fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig1)

# Moving Averages for 100 days
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()

fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Moving Averages for 200 days
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()

fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig3)


x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale
y = y*scale

# Original vs Predicted Price
st.subheader('Original Price vs Predicted Price')

fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)


# Forecast next 100 days
future_input = data_test_scale[-100:]
future_input = future_input.reshape(1, future_input.shape[0], future_input.shape[1])

future_preds = []
for _ in range(100):
    future_price = model.predict(future_input, verbose=0)  # shape (1,1)
    future_preds.append(future_price[0, 0])

    # reshape properly and append
    new_val = np.array([[[future_price[0, 0]]]])  # shape (1,1,1)
    future_input = np.append(future_input[:, 1:, :], new_val, axis=1)

future_preds = np.array(future_preds)
future_preds = future_preds * scale

# Original vs Predicted Price + Future Forecast
st.subheader('Original Price vs Predicted Price + 100 Days Forecast')

fig5 = plt.figure(figsize=(10, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.plot(range(len(y), len(y) + 100), future_preds, 'b', label='Future 100 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)