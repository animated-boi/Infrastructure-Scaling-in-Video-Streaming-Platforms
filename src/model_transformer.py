import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input, Dense, Dropout, Add, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.models import Model

# -------------------------------
def create_dataset(series, look_back=7):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)



def build_transformer_model(input_shape, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    
    # Step 1: Expand input feature dimension
    x = Dense(d_model)(inputs)

    # Step 2: Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([x, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Step 3: Feedforward
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(d_model)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

    # Step 4: Pooling and Output
    pooled = GlobalAveragePooling1D()(ff_output)
    outputs = Dense(1)(pooled)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model







# -------------------------------
if __name__ == '__main__':
    print("🔁 Starting Transformer Forecasting...")

    # Load dataset
    df = pd.read_csv('results/aggregated_timeseries.csv')
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    df_ent = df[df['category_name'] == 'Entertainment'].copy().sort_values('trending_date')
    series = df_ent['views'].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    # Create supervised samples
    look_back = 7
    X, y = create_dataset(series_scaled, look_back=look_back)

    # Reshape for attention: (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = build_transformer_model(input_shape=(look_back, 1))
    model.summary()

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=50, batch_size=8, verbose=1)

    # Predict
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # RMSE
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print("Transformer Test RMSE:", rmse)

    # Save results
    test_dates = df_ent['trending_date'].values[split + look_back: split + look_back + len(y_test)]
    forecast_df = pd.DataFrame({
        'date': test_dates,
        'actual_views': y_test_inv.flatten(),
        'predicted_views': y_pred_inv.flatten()
    })
    forecast_df.to_csv('results/transformer_forecast.csv', index=False)
    print("Transformer forecast saved to results/transformer_forecast.csv")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(test_dates, y_test_inv, label='Actual Views')
    plt.plot(test_dates, y_pred_inv, label='Predicted Views', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Views')
    plt.title('Transformer Forecast vs Actual (Entertainment)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
