# src/model_cnn_lstm.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Load dataset
print("📦 Loading dataset...")
df = pd.read_csv("results/aggregated_timeseries.csv")
df = df.groupby("trending_date")["views"].sum().reset_index()
df["trending_date"] = pd.to_datetime(df["trending_date"])
df = df.sort_values("trending_date")

# Normalize views
scaler = MinMaxScaler()
df["views"] = scaler.fit_transform(df["views"].values.reshape(-1, 1))

# Prepare supervised learning dataset
look_back = 7
X, y, dates = [], [], []
for i in range(len(df) - look_back):
    X.append(df["views"].iloc[i:i+look_back].values)
    y.append(df["views"].iloc[i+look_back])
    dates.append(df["trending_date"].iloc[i+look_back])

X, y = np.array(X), np.array(y)
X = np.expand_dims(X, axis=2)  # Shape: (samples, timesteps, 1)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = dates[split:]

# Build CNN-LSTM model
print("🧠 Building CNN-LSTM model...")
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, 1)),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(loss="mse", optimizer="adam")
model.summary()

# Train
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict and inverse scale
y_pred = model.predict(X_test).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
print(f"📉 CNN-LSTM RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

# Save forecast
forecast_df = pd.DataFrame({
    "date": dates_test,
    "actual_views": y_test_inv,
    "predicted_views": y_pred_inv
})
forecast_df.to_csv("results/cnn_lstm_forecast.csv", index=False)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test_inv, label="Actual Views")
plt.plot(dates_test, y_pred_inv, label="Predicted Views")
plt.title("CNN-LSTM Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Views")
plt.legend()
plt.tight_layout()
plt.savefig("figures/Figure_CNN_LSTM_Forecast.png")
plt.show()
