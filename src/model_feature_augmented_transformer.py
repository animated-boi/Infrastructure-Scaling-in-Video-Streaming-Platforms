import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Add
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from keras.optimizers import Adam
import tensorflow as tf

print("🔁 Starting Feature-Augmented Transformer Forecast...")

# Load base timeseries
df = pd.read_csv("results/aggregated_timeseries.csv")
df["trending_date"] = pd.to_datetime(df["trending_date"])
df.sort_values("trending_date", inplace=True)

# Load ARIMA and Prophet forecasts
arima = pd.read_csv("results/arima_forecast.csv")
prophet = pd.read_csv("results/prophet_forecast.csv")

arima["date"] = pd.to_datetime(arima["date"])
prophet["ds"] = pd.to_datetime(prophet["ds"])

# Align forecasts with main df
df = df.merge(arima[["date", "predicted_views"]].rename(columns={"date": "trending_date", "predicted_views": "arima_forecast"}), on="trending_date")
df = df.merge(prophet[["ds", "predicted_views"]].rename(columns={"ds": "trending_date", "predicted_views": "prophet_forecast"}), on="trending_date")

# Drop NA (if any)
df.dropna(inplace=True)

# Features
features = ["views", "avg_likes", "num_trending_videos", "arima_forecast", "prophet_forecast"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Create sequences (look_back = 7)
look_back = 7
X, y = [], []
for i in range(len(df) - look_back):
    X.append(df[features].iloc[i:i+look_back].values)
    y.append(df["views"].iloc[i+look_back])
X, y = np.array(X), np.array(y)

# Train/Test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Build Transformer model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64)(inputs)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    ff = Dense(128, activation="relu")(x)
    ff = Dense(64)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

model = build_transformer_model((look_back, len(features)))
model.summary()
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print(f"📉 Feature-Augmented Transformer RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

# Save results
forecast_df = pd.DataFrame({
    "date": df["trending_date"].iloc[-len(y_test):].values,
    "actual_views": y_test,
    "predicted_views": y_pred.flatten()
})
forecast_df.to_csv("results/feature_aug_transformer_forecast.csv", index=False)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(forecast_df["date"], forecast_df["actual_views"], label="Actual Views")
plt.plot(forecast_df["date"], forecast_df["predicted_views"], label="Predicted Views")
plt.legend()
plt.title("Feature-Augmented Transformer Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Views")
plt.tight_layout()
plt.savefig("figures/Figure_Feature_Augmented_Transformer.png")
plt.show()
