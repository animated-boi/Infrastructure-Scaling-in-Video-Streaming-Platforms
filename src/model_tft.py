import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_lightning import seed_everything, Trainer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError

# -------------------------
# Load processed data
df = pd.read_csv("results/tft_input_data.csv")
df["trending_date"] = pd.to_datetime(df["trending_date"])

# FIX: Ensure the target column ("views") is float
df["views"] = df["views"].astype(float)

# -------------------------
# Set constants
max_encoder_length = 30
max_prediction_length = 7

# -------------------------
# Define training TimeSeriesDataSet
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx < x.time_idx.max() - max_prediction_length],
    time_idx="time_idx",
    target="views",
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["avg_likes_scaled", "num_trending_videos_scaled"],
    time_varying_unknown_reals=["views"],
    time_varying_known_categoricals=["day_of_week"],
    static_categoricals=["series_id"],
    target_normalizer=GroupNormalizer(
        groups=["series_id"],
        transformation="softplus"
    )
)

# -------------------------
# Create validation set from training dataset
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
train_dataloader = training.to_dataloader(train=True, batch_size=16)
val_dataloader = validation.to_dataloader(train=False, batch_size=16)

# -------------------------
# Define and train TFT model
seed_everything(42)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=MeanSquaredError(),  # (this argument is typically 'RMSE()' or another loss function; adjust accordingly)
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=4,
    logging_metrics=[MeanAbsolutePercentageError()]
)

trainer = Trainer(max_epochs=30, gradient_clip_val=0.1)
from pytorch_forecasting.models.base_model import BaseModelWithCovariates

lightning_model = tft.to_lightning_module()
trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)




# -------------------------
# Predict on validation set
best_model = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_model.predict(val_dataloader)

# -------------------------
# Calculate RMSE
rmse = mean_squared_error(actuals.numpy(), predictions.numpy(), squared=False)
print(f"📈 TFT Test RMSE: {rmse:,.2f}")

# -------------------------
# Plot predictions
dates = df.loc[df.time_idx >= df.time_idx.max() - max_prediction_length, "trending_date"]
plt.figure(figsize=(10, 5))
plt.plot(dates, actuals.numpy(), label="Actual")
plt.plot(dates, predictions.numpy(), label="Predicted")
plt.title("TFT Forecast vs Actual (Entertainment)")
plt.xlabel("Date")
plt.ylabel("Views")
plt.legend()
plt.grid()
os.makedirs("results", exist_ok=True)
plt.savefig("results/Figure_TFT_ForecastVsActual.png")
plt.show()
