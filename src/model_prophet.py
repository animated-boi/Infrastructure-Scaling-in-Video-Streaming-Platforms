import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def forecast_with_prophet(input_csv, category_name, forecast_days=30):
    # Load aggregated time series
    df = pd.read_csv(input_csv)

    # Filter for one category (e.g., Entertainment)
    df_category = df[df['category_name'] == category_name].copy()

    # Prepare data for Prophet
    df_category = df_category[['trending_date', 'views']]
    df_category.rename(columns={'trending_date': 'ds', 'views': 'y'}, inplace=True)

    # Ensure datetime format
    df_category['ds'] = pd.to_datetime(df_category['ds'])

    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df_category)

    # Create future DataFrame
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f"Prophet Forecast for '{category_name}' Views")
    plt.xlabel("Date")
    plt.ylabel("Views")
    plt.tight_layout()
    plt.show()

    # Save forecasted data
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('results/prophet_forecast.csv', index=False)
    print("Forecast saved to 'results/prophet_forecast.csv'")


if __name__ == '__main__':
    INPUT_CSV = 'results/aggregated_timeseries.csv'
    CATEGORY = 'Entertainment'  # Change if needed
    forecast_with_prophet(INPUT_CSV, CATEGORY)
