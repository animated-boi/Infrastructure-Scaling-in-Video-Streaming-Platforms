# import pandas as pd
# import numpy as np
# import os
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# import matplotlib.pyplot as plt

# forecast_files = {
#     "Prophet": "results/prophet_forecast.csv",
#     "ARIMA": "results/arima_forecast.csv",
#     "LSTM_Univariate": "results/lstm_final_forecast.csv",
#     "Transformer_Univariate": "results/transformer_forecast.csv",
#     "LSTM_Multivariate": "results/lstm_mv_forecast.csv",
#     "Transformer_Multivariate": "results/transformer_mv_forecast.csv"
# }

# actual_df = pd.read_csv("results/aggregated_timeseries.csv")
# actual_df = actual_df.groupby("trending_date")["views"].sum().reset_index()
# actual_df["trending_date"] = pd.to_datetime(actual_df["trending_date"])

# results = []

# print("\n📊 Starting comprehensive evaluation of all models...")

# for model_name, file_path in forecast_files.items():
#     if not os.path.exists(file_path):
#         print(f"❌ Missing forecast file for {model_name}")
#         continue

#     forecast_df = pd.read_csv(file_path)
#     if "trending_date" not in forecast_df.columns: # Safely generate date range aligned to forecast length 
#         if "trending_date" in actual_df.columns and len(actual_df) >= len(forecast_df): 
#             start_date = pd.to_datetime(actual_df["trending_date"].iloc[-len(forecast_df)]) 
            
#         else: 
#             start_date = pd.to_datetime("2018-05-01") # fallback default 
        
#         forecast_df["trending_date"] = pd.date_range( start=start_date, periods=len(forecast_df), freq="D" ) 
        
#     else: 
#         forecast_df["trending_date"] = pd.to_datetime(forecast_df["trending_date"])

#     if "predicted_views" not in forecast_df.columns:
#         print(f"❌ Failed to evaluate {model_name}: 'predicted_views' not found")
#         continue

#     merged_df = pd.merge(actual_df, forecast_df, on="trending_date", how="inner")
#     # print("📄 Columns in merged_df:", merged_df.columns.tolist())
#     # print("🔢 First few rows:\n", merged_df.head())
#     if "actual_views" in merged_df.columns:
#         y_true = merged_df["actual_views"].values
#     elif "views" in merged_df.columns:
#         y_true = merged_df["views"].values
#     elif "views_y" in merged_df.columns:
#         y_true = merged_df["views_y"].values
#     else:
#         raise KeyError("Could not find actual view column in merged_df")
    
#     y_pred = merged_df["predicted_views"].values

#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = mean_absolute_percentage_error(y_true, y_pred)

#     results.append({
#         "Model": model_name,
#         "RMSE": rmse,
#         "MAE": mae,
#         "MAPE": mape * 100
#     })

# results_df = pd.DataFrame(results)
# results_df.sort_values("RMSE", inplace=True)
# results_df.to_csv("results/model_evaluation_summary.csv", index=False)

# # Plot RMSE, MAE, MAPE
# plt.figure(figsize=(10, 6))
# plt.bar(results_df["Model"], results_df["RMSE"])
# plt.xticks(rotation=45)
# plt.title("Model Comparison - RMSE")
# plt.ylabel("RMSE")
# plt.tight_layout()
# plt.savefig("figures/Figure_Model_Comparison_RMSE.png")
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.bar(results_df["Model"], results_df["MAE"])
# plt.xticks(rotation=45)
# plt.title("Model Comparison - MAE")
# plt.ylabel("MAE")
# plt.tight_layout()
# plt.savefig("figures/Figure_Model_Comparison_MAE.png")
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.bar(results_df["Model"], results_df["MAPE"])
# plt.xticks(rotation=45)
# plt.title("Model Comparison - MAPE")
# plt.ylabel("MAPE (%)")
# plt.tight_layout()
# plt.savefig("figures/Figure_Model_Comparison_MAPE.png")
# plt.show()

# print("✅ Evaluation complete. Results saved to results/model_evaluation_summary.csv")


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

forecast_files = {
    "Prophet": "results/prophet_forecast.csv",
    "ARIMA": "results/arima_forecast.csv",
    "LSTM_Univariate": "results/lstm_final_forecast.csv",
    "Transformer_Univariate": "results/transformer_forecast.csv",
    "LSTM_Multivariate": "results/lstm_mv_forecast.csv",
    "Transformer_Multivariate": "results/transformer_mv_forecast.csv",
    "FeatureAug_Transformer": "results/feature_aug_transformer_forecast.csv",
    "FeatureAug_LSTM": "results/feature_aug_lstm_forecast.csv",
    "CNN_LSTM": "results/cnn_lstm_forecast.csv",
    "Hybrid_Ensemble" : "results/hybrid_forecast.csv"
}

actual_df = pd.read_csv("results/aggregated_timeseries.csv")
actual_df = actual_df.groupby("trending_date")["views"].sum().reset_index()
actual_df["trending_date"] = pd.to_datetime(actual_df["trending_date"])

results = []

print("\n📊 Starting comprehensive evaluation of all models...")

for model_name, file_path in forecast_files.items():
    if not os.path.exists(file_path):
        print(f"❌ Missing forecast file for {model_name}")
        continue

    forecast_df = pd.read_csv(file_path)

    # Ensure correct datetime alignment
    if "trending_date" not in forecast_df.columns:
        if "ds" in forecast_df.columns:
            forecast_df.rename(columns={"ds": "trending_date"}, inplace=True)
        elif "date" in forecast_df.columns:
            forecast_df.rename(columns={"date": "trending_date"}, inplace=True)
        else:
            start_date = pd.to_datetime(actual_df["trending_date"].iloc[-len(forecast_df)])
            forecast_df["trending_date"] = pd.date_range(start=start_date, periods=len(forecast_df), freq="D")

    forecast_df["trending_date"] = pd.to_datetime(forecast_df["trending_date"])

    if "predicted_views" not in forecast_df.columns:
        print(f"❌ Failed to evaluate {model_name}: 'predicted_views' not found")
        continue

    merged_df = pd.merge(actual_df, forecast_df, on="trending_date", how="inner")

    if "actual_views" in merged_df.columns:
        y_true = merged_df["actual_views"].values
    elif "views" in merged_df.columns:
        y_true = merged_df["views"].values
    elif "views_y" in merged_df.columns:
        y_true = merged_df["views_y"].values
    else:
        raise KeyError(f"❌ Could not find actual view column in merged_df for {model_name}")

    y_pred = merged_df["predicted_views"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape * 100
    })

results_df = pd.DataFrame(results)
results_df.sort_values("RMSE", inplace=True)
results_df.to_csv("results/model_evaluation_summary.csv", index=False)

# Plot RMSE, MAE, MAPE
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.xticks(rotation=45)
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("figures/Figure_Model_Comparison_RMSE.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["MAE"])
plt.xticks(rotation=45)
plt.title("Model Comparison - MAE")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("figures/Figure_Model_Comparison_MAE.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["MAPE"])
plt.xticks(rotation=45)
plt.title("Model Comparison - MAPE")
plt.ylabel("MAPE (%)")
plt.tight_layout()
plt.savefig("figures/Figure_Model_Comparison_MAPE.png")
plt.show()

print("✅ Evaluation complete. Results saved to results/model_evaluation_summary.csv")
