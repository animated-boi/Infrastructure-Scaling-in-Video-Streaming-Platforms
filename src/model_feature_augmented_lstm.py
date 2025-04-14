# src/model_feature_augmented_lstm.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

print("🔁 Starting Feature-Augmented LSTM Forecast...")

# Load main dataset
df = pd.read_csv("results/aggregated_timeseries.csv")
df["trending_date"] = pd.to_datetime(df["trending_date"])
df.sort_values("trending_date", inplace=True)

# Load ARIMA and Prophet forecasts
arima = pd.read_csv("results/arima_forecast.csv")
prophet = pd.read_csv("results/prophet_forecast.csv")
arima["date"] = pd.to_datetime(arima["date"])
prophet["ds"] = pd.to_datetime(prophet["ds"])

# Merge forecasts with base
df = df.merge(arima[["date", "predicted_views"]].rename(columns={"date": "trending_date", "predicted_views": "arima_forecast"}), on="trending_date")
df = df.merge(prophet[["ds", "predicted_views"]].rename(columns={"ds": "trending_date", "predicted_views": "prophet_forecast"}), on="trending_date")

df.dropna(inplace=True)

# Feature scaling
features = ["views", "avg_likes", "num_trending_videos", "arima_forecast", "prophet_forecast"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Sequence creation
look_back = 7
X, y = [], []
for i in range(len(df) - look_back):
    X.append(df[features].iloc[i:i+look_back].values)
    y.append(df["views"].iloc[i+look_back])
X, y = np.array(X), np.array(y)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"X shape: {X.shape}, y shape: {y.shape}")

# LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(look_back, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.summary()
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print(f"📉 Feature-Augmented LSTM RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

# Save forecast
forecast_df = pd.DataFrame({
    "date": df["trending_date"].iloc[-len(y_test):].values,
    "actual_views": y_test,
    "predicted_views": y_pred.flatten()
})
forecast_df.to_csv("results/feature_aug_lstm_forecast.csv", index=False)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(forecast_df["date"], forecast_df["actual_views"], label="Actual Views")
plt.plot(forecast_df["date"], forecast_df["predicted_views"], label="Predicted Views")
plt.legend()
plt.title("Feature-Augmented LSTM Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Views")
plt.tight_layout()
plt.savefig("figures/Figure_Feature_Augmented_LSTM.png")
plt.show()
