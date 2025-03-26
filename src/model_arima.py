# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA


# def forecast_with_arima(input_csv, category_name, forecast_days=30):
#     # Load aggregated time series
#     df = pd.read_csv(input_csv)

#     # Filter for one category (e.g., Entertainment)
#     df_category = df[df['category_name'] == category_name].copy()

#     # Convert trending_date to datetime
#     df_category['trending_date'] = pd.to_datetime(df_category['trending_date'])
#     df_category.sort_values('trending_date', inplace=True)

#     # Set date as index and select views column
#     ts = df_category.set_index('trending_date')['views']

#     # Fit ARIMA model (order can be tuned later)
#     model = ARIMA(ts, order=(5, 1, 2))
#     fitted_model = model.fit()

#     # Forecast future values
#     forecast = fitted_model.forecast(steps=forecast_days)

#     # Plot original and forecasted values
#     plt.figure(figsize=(10, 5))
#     plt.plot(ts, label='Historical Views')
#     plt.plot(pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days), forecast, label='Forecast', linestyle='--')
#     plt.title(f"ARIMA Forecast for '{category_name}' Views")
#     plt.xlabel("Date")
#     plt.ylabel("Views")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # Save forecast to CSV
#     forecast_df = pd.DataFrame({
#         'date': pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
#         'forecast_views': forecast
#     })
#     forecast_df.to_csv('results/arima_forecast.csv', index=False)
#     print("Forecast saved to 'results/arima_forecast.csv'")


# if __name__ == '__main__':
#     INPUT_CSV = 'results/aggregated_timeseries.csv'
#     CATEGORY = 'Entertainment'  # Change if needed
#     forecast_with_arima(INPUT_CSV, CATEGORY)
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load aggregated data
df = pd.read_csv('results/aggregated_timeseries.csv')
df['trending_date'] = pd.to_datetime(df['trending_date'])

# Filter for Entertainment category
df_ent = df[df['category_name'] == 'Entertainment']
df_ent = df_ent.set_index('trending_date')
series = df_ent['views']

# Fit ARIMA model on historical data
model = ARIMA(series, order=(5, 1, 2))
model_fit = model.fit()

# Predict in-sample values (same length as actual data)
start_date = series.index[0]
end_date = series.index[-1]
pred = model_fit.predict(start=0, end=len(series)-1, typ='levels')
pred.index = series.index

# Save predictions
forecast_df = pd.DataFrame({
    'date': pred.index,
    'forecast_views': pred.values
})
forecast_df.to_csv('results/arima_forecast.csv', index=False)

# Plot comparison
plt.figure(figsize=(10, 4))
plt.plot(series, label='Actual')
plt.plot(pred, label='ARIMA Forecast (In-Sample)', linestyle='--')
plt.legend()
plt.title('ARIMA In-Sample Forecast vs Actual')
plt.tight_layout()
plt.show()
