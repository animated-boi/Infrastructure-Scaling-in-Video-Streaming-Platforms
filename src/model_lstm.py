# src/model_lstm.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_dataset(series, look_back=7):
    """
    Converts a time series into a supervised learning dataset.
    Each sample will have 'look_back' time steps as input and the next time step as output.
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back)])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # Load aggregated time series data
    df = pd.read_csv('results/aggregated_timeseries.csv')
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    
    # Filter for the 'Entertainment' category and sort by date
    df_ent = df[df['category_name'] == 'Entertainment'].copy().sort_values('trending_date')
    
    # Use the 'views' column as our time series
    series = df_ent['views'].values.reshape(-1, 1)
    
    # Normalize the data to the range [0, 1] using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series)
    
    # Define the window length (number of past days to use as input)
    look_back = 7
    
    # Create supervised learning dataset using a sliding window
    X, y = create_dataset(series_scaled, look_back=look_back)
    
    # Report the shapes of the input (X) and output (y) datasets
    print("Shape of input dataset X:", X.shape)
    print("Shape of output dataset y:", y.shape)


# 2: Split the dataset into training and test sets (80% training, 20% test)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Training set shape - X:", X_train.shape, " y:", y_train.shape)
print("Test set shape - X:", X_test.shape, " y:", y_test.shape)

# 3: Build and compile the LSTM model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')


# 4: Train the model

history = model.fit(X_train, y_train,
                    epochs=50, 
                    batch_size=8,
                    validation_data=(X_test, y_test),
                    verbose=1)


# 5: Make predictions on the test set

y_pred = model.predict(X_test)

# 6: Inverse transform to get predictions and actual values back to original scale

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate RMSE for the test set
rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"LSTM Test RMSE: {rmse:.2f}")


# 7: Save predictions to CSV and plot the results

test_dates = df_ent['trending_date'].values[split_index + 7 : split_index + 7 + len(y_test)]

forecast_df = pd.DataFrame({
    'date': test_dates,
    'actual_views': y_test_inv.flatten(),
    'predicted_views': y_pred_inv.flatten()
})
forecast_df.to_csv('results/lstm_forecast.csv', index=False)
print("LSTM forecast saved to results/lstm_forecast.csv")

# Plot the actual vs. predicted views
plt.figure(figsize=(10, 4))
plt.plot(test_dates, y_test_inv, label='Actual Views')
plt.plot(test_dates, y_pred_inv, label='Predicted Views', linestyle='--')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Views')
plt.title('LSTM Forecast vs Actual (Entertainment Category)')
plt.legend()
plt.tight_layout()
plt.show()