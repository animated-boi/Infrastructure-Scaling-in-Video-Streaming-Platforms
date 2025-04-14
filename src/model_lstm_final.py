# src/model_lstm_final.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_dataset(series, look_back=7):
    """
    Converts a time series into a supervised learning dataset.
    Each sample consists of 'look_back' time steps as input with the next time step as the target.
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back)])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # -------------------------------
    # Data Preparation
    df = pd.read_csv('results/aggregated_timeseries.csv')
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    
    # Filter the data for 'Entertainment' category and sort by date
    df_ent = df[df['category_name'] == 'Entertainment'].copy().sort_values('trending_date')
    print("Total samples in Entertainment category:", len(df_ent))
    
    # Use the 'views' column as the univariate time series and reshape for the scaler
    series = df_ent['views'].values.reshape(-1, 1)
    
    # Normalize the data to [0,1] using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series)
    
    # Create supervised dataset using a sliding window (look_back = 7)
    look_back = 7
    X, y = create_dataset(series_scaled, look_back=look_back)
    print("Shape of input dataset X:", X.shape)
    print("Shape of output dataset y:", y.shape)
    
    # -------------------------------
    # Split into training and test sets (80% training, 20% test)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print("Training set shape - X:", X_train.shape, " y:", y_train.shape)
    print("Test set shape - X:", X_test.shape, " y:", y_test.shape)
    
    # -------------------------------
    # Build the LSTM model with the tuned hyperparameters:
    # - 2 LSTM layers
    # - 64 units per layer
    # - 20% dropout after the first LSTM layer
    model = Sequential()
    # First LSTM layer with 'return_sequences=True' because we're stacking another LSTM layer
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    # Second LSTM layer
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    # -------------------------------
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)
    
    # -------------------------------
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values back to original scale
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Compute RMSE on the test set (in the original scale)
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print("Final LSTM Model Test RMSE:", rmse)
    
    # -------------------------------
    # Save the forecasted results to a CSV file
    # The corresponding test dates are taken from df_ent; note the offset due to the look-back window
    test_dates = df_ent['trending_date'].values[split_index + look_back: split_index + look_back + len(y_test)]
    forecast_df = pd.DataFrame({
        'date': test_dates,
        'actual_views': y_test_inv.flatten(),
        'predicted_views': y_pred_inv.flatten()
    })
    forecast_df.to_csv('results/lstm_final_forecast.csv', index=False)
    print("Final LSTM forecast saved to results/lstm_final_forecast.csv")
    
    # -------------------------------
    # Plot Actual vs Predicted Views
    plt.figure(figsize=(10, 4))
    plt.plot(test_dates, y_test_inv, label='Actual Views')
    plt.plot(test_dates, y_pred_inv, label='Predicted Views', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Views')
    plt.title('Final LSTM Forecast vs Actual (Entertainment Category)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
