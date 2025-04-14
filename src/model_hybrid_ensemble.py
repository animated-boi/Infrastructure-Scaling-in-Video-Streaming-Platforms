import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


print("🔧 Starting Hybrid Ensemble Forecast...")

# Load forecasts
df_lstm = pd.read_csv("results/lstm_final_forecast.csv")
df_transformer = pd.read_csv("results/transformer_forecast.csv")
df_arima = pd.read_csv("results/arima_forecast.csv")
df_prophet = pd.read_csv("results/prophet_forecast.csv")
df_actual = pd.read_csv("results/aggregated_timeseries.csv")

# Ensure date alignment

df_lstm["date"] = pd.to_datetime(df_lstm["date"])
df_transformer["date"] = pd.to_datetime(df_transformer["date"])
df_arima["date"] = pd.to_datetime(df_arima["date"])
df_prophet.rename(columns={"ds": "date"}, inplace=True)
df_prophet["date"] = pd.to_datetime(df_prophet["date"])
df_actual["trending_date"] = pd.to_datetime(df_actual["trending_date"])

# Get actual view series for the last N days (same length as predictions)
num_days = len(df_lstm)
df_actual_recent = df_actual.tail(num_days)[["trending_date", "views"]].copy()
df_actual_recent.rename(columns={"trending_date": "date", "views": "actual_views"}, inplace=True)

# Merge all predictions by date
merged = df_actual_recent.merge(df_lstm[["date", "predicted_views"]], on="date")
merged.rename(columns={"predicted_views": "lstm"}, inplace=True)
merged = merged.merge(df_transformer[["date", "predicted_views"]], on="date")
merged.rename(columns={"predicted_views": "transformer"}, inplace=True)
merged = merged.merge(df_arima[["date", "predicted_views"]], on="date")
merged.rename(columns={"predicted_views": "arima"}, inplace=True)
merged = merged.merge(df_prophet[["date", "predicted_views"]], on="date")
merged.rename(columns={"predicted_views": "prophet"}, inplace=True)

# Create hybrid forecast
merged["hybrid_forecast"] = merged[["lstm", "transformer", "arima", "prophet"]].mean(axis=1)

# Save hybrid forecast
hybrid_df = merged[["date", "actual_views", "hybrid_forecast"]]
hybrid_df.rename(columns={"hybrid_forecast": "predicted_views"}, inplace=True)
hybrid_df.to_csv("results/hybrid_forecast.csv", index=False)
print("✅ Hybrid forecast saved to results/hybrid_forecast.csv")

# Evaluate
rmse = np.sqrt(mean_squared_error(hybrid_df["actual_views"], hybrid_df["predicted_views"]))
mae = mean_absolute_error(hybrid_df["actual_views"], hybrid_df["predicted_views"])
mape = mean_absolute_percentage_error(hybrid_df["actual_views"], hybrid_df["predicted_views"]) * 100
print(f"📉 Hybrid RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

# Plot
plt.figure(figsize=(10,5))
plt.plot(hybrid_df["date"], hybrid_df["actual_views"], label="Actual Views")
plt.plot(hybrid_df["date"], hybrid_df["predicted_views"], label="Hybrid Forecast")
plt.xlabel("Date")
plt.ylabel("Views")
plt.title("Hybrid Forecast vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig("figures/Figure_Hybrid_Ensemble_Forecast.png")
plt.show()
print("Forecast plot saved to figures/Figure_Hybrid_Ensemble_Forecast.png")
