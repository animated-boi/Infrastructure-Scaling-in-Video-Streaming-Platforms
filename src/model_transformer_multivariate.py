# src/model_transformer_multivariate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models

print("🔁 Starting Multivariate Transformer Forecasting...")

# Load dataset
df = pd.read_csv("results/aggregated_timeseries.csv")
df['trending_date'] = pd.to_datetime(df['trending_date'])
df = df.sort_values('trending_date')
df['day_of_week'] = df['trending_date'].dt.dayofweek

features = ['avg_likes', 'num_trending_videos', 'day_of_week']
target = 'views'

# Normalize features and target
scalers = {}
scaled_cols = []
for col in features + [target]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[col]])
    scalers[col] = scaler
    scaled_cols.append(scaled)

scaled_data = np.hstack(scaled_cols)
look_back = 7

X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i+look_back, :-1])  # all features except target
    y.append(scaled_data[i+look_back, -1])     # only target

X, y = np.array(X), np.array(y)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train-test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build Transformer model
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(64)(inputs)

    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(64)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model(input_shape=(look_back, X.shape[2]))
model.summary()

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Predict
y_pred = model.predict(X_test)
y_pred_inv = scalers[target].inverse_transform(y_pred)
y_test_inv = scalers[target].inverse_transform(y_test.reshape(-1, 1))

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"Multivariate Transformer RMSE: {rmse:.2f}")

# Save forecast
dates = df['trending_date'].iloc[look_back + split_idx:]
forecast_df = pd.DataFrame({
    'trending_date': dates,
    'actual_views': y_test_inv.flatten(),
    'predicted_views': y_pred_inv.flatten()
})
forecast_df.to_csv("results/transformer_mv_forecast.csv", index=False)
print("Forecast saved to results/transformer_mv_forecast.csv")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['trending_date'], forecast_df['actual_views'], label="Actual Views")
plt.plot(forecast_df['trending_date'], forecast_df['predicted_views'], label="Predicted Views")
plt.title("Multivariate Transformer Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Views")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/Figure_Transformer_Multivariate_Forecast.png")
plt.show()
