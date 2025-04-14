# src/model_lstm_tuning.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt

# Helper function: Create supervised dataset using a sliding window
def create_dataset(series, look_back=7):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back)])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

# -------------------------------
# Data Preparation

# Load the aggregated time series data
df = pd.read_csv('results/aggregated_timeseries.csv')
df['trending_date'] = pd.to_datetime(df['trending_date'])
# Filter for Entertainment category and sort by date
df_ent = df[df['category_name'] == 'Entertainment'].copy().sort_values('trending_date')

# Use the 'views' column as our univariate time series and reshape it for the scaler
series = df_ent['views'].values.reshape(-1, 1)

# Normalize the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

# Define the look-back window and create supervised dataset
look_back = 7
X, y = create_dataset(series_scaled, look_back=look_back)

# Split the dataset into training and testing (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Shape of input dataset X:", X.shape)
print("Shape of output dataset y:", y.shape)
print("Training set shape - X:", X_train.shape, " y:", y_train.shape)
print("Test set shape - X:", X_test.shape, " y:", y_test.shape)

# -------------------------------
# Define Hypermodel using Keras Tuner
def build_model(hp):
    model = Sequential()
    # Tune the number of LSTM layers: choice between 1 or 2 layers
    num_layers = hp.Choice('num_layers', [1, 2], default=1)
    # Tune the number of units in each LSTM layer
    units = hp.Int('units', min_value=32, max_value=128, step=32, default=64)
    # Tune the dropout rate (to help prevent overfitting)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    
    # Build model architecture based on the hyperparameters
    if num_layers == 1:
        model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    else:  # Use two LSTM layers
        model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units, activation='relu'))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# -------------------------------
# Set up the Keras Tuner

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # You can increase this for a more exhaustive search
    executions_per_trial=1,
    directory='lstm_tuning_dir',
    project_name='lstm_forecast_tuning'
)

print("Starting hyperparameter tuning...")

tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)

# Retrieve the best hyperparameters from the search
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters found:")
print(f"Number of LSTM layers: {best_hps.get('num_layers')}")
print(f"Units per layer: {best_hps.get('units')}")
print(f"Dropout rate: {best_hps.get('dropout_rate')}")
