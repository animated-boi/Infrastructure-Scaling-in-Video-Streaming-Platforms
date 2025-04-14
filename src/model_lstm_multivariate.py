# src/model_lstm_multivariate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

print("🔄 Loading and preparing multivariate dataset...")

# Load data
df = pd.read_csv("results/aggregated_timeseries.csv")
df['trending_date'] = pd.to_datetime(df['trending_date'])
df = df.sort_values('trending_date')

# Feature engineering
df['day_of_week'] = df['trending_date'].dt.dayofweek  # 0 = Monday

# Select features and target
features = ['avg_likes', 'num_trending_videos', 'day_of_week']
target = 'views'

# Scale features and target
scalers = {}
scaled_features = []
for col in features + [target]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[col]])
    scalers[col] = scaler
    scaled_features.append(scaled)

scaled_data = np.hstack(scaled_features)

# Hyperparameters
look_back = 7

# Create sequences (X: 7-day window, y: next day views)
X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i+look_back, :-1])  # All features except views
    y.append(scaled_data[i+look_back, -1])     # Only views as target

X, y = np.array(X), np.array(y)

print(f"Shape of input X: {X.shape}")
print(f"Shape of target y: {y.shape}")

# Train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(look_back, X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

print("🧠 Training LSTM model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Predict
y_pred = model.predict(X_test)

# Inverse scale predictions and actuals
y_pred_inv = scalers[target].inverse_transform(y_pred)
y_test_inv = scalers[target].inverse_transform(y_test.reshape(-1, 1))

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"📉 Multivariate LSTM RMSE: {rmse:.2f}")

# Save forecast
forecast_df = df.iloc[look_back + split_idx:].copy()
forecast_df['predicted_views'] = y_pred_inv
forecast_df[['trending_date', 'category_name', 'views', 'predicted_views']].to_csv("results/lstm_mv_forecast.csv", index=False)
print("✅ Forecast saved to results/lstm_mv_forecast.csv")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['trending_date'], forecast_df['views'], label="Actual Views")
plt.plot(forecast_df['trending_date'], forecast_df['predicted_views'], label="Predicted Views")
plt.title("Multivariate LSTM Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Views")
plt.legend()
plt.grid(True)
plt.tight_layout()
\
plt.show()

print("End")
