import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load file
df = pd.read_csv("results/aggregated_timeseries.csv", parse_dates=["trending_date"])

# Filter for Entertainment
df = df[df["category_name"] == "Entertainment"].copy()
df.sort_values("trending_date", inplace=True)

# Add time index
df["time_idx"] = np.arange(len(df))

# Add day-of-week
df["day_of_week"] = df["trending_date"].dt.day_name()  # e.g., "Monday", "Tuesday"


# Add series ID
df["series_id"] = "entertainment"

# Scale numeric covariates
scaler = StandardScaler()
df["avg_likes_scaled"] = scaler.fit_transform(df[["avg_likes"]])
df["num_trending_videos_scaled"] = scaler.fit_transform(df[["num_trending_videos"]])

# Save processed dataset
df.to_csv("results/tft_input_data.csv", index=False)
print("✅ Multivariate TFT dataset saved to results/tft_input_data.csv")
