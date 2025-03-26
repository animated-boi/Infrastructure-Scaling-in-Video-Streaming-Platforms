import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(pred_path, actual_series, model_name, col_name):
    forecast_df = pd.read_csv(pred_path)

    # Use proper datetime column and value column
    forecast_df['date'] = pd.to_datetime(forecast_df.iloc[:, 0])
    forecast_df.set_index('date', inplace=True)

    # Align dates between actual and forecast
    common_dates = forecast_df.index.intersection(actual_series.index)
    actual = actual_series.loc[common_dates]
    predicted = forecast_df.loc[common_dates][col_name]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)

    return {
        'Model': model_name,
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE': round(mape, 2)
    }

if __name__ == '__main__':
    # Load actuals
    df = pd.read_csv('results/aggregated_timeseries.csv')
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    actual_ts = df[df['category_name'] == 'Entertainment'].copy()
    actual_ts.set_index('trending_date', inplace=True)
    actual_series = actual_ts['views']

    # Evaluate Prophet (uses 'yhat')
    prophet_result = evaluate_model('results/prophet_forecast.csv', actual_series, 'Prophet', 'yhat')

    # Evaluate ARIMA (uses 'forecast_views')
    arima_result = evaluate_model('results/arima_forecast.csv', actual_series, 'ARIMA', 'forecast_views')

    # Print results
    print("\n📊 Model Performance Comparison:")
    print(pd.DataFrame([prophet_result, arima_result]))
