import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the LSTM_Univariate forecast
df = pd.read_csv("results/lstm_final_forecast.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Forecasted views
views = df["predicted_views"].values

# Parameters
unit_capacity = 50_000_000   # 1 unit handles 50 million views
buffer_units = 0.5             # Allow only 0.5 extra units beyond what is needed
max_scaling_cap = 1_000_000_000  # Optional upper cap for realistic scaling

allocated = []
for v in views:
    # Required units (rounded up)
    required_units = int((v + unit_capacity - 1) // unit_capacity)

    # Allocated = required + buffer (up to 1 extra units)
    allocated_units = required_units + buffer_units

    # Optional: avoid over-allocating past max cap
    allocated_capacity = min(allocated_units * unit_capacity, max_scaling_cap)

    allocated.append(allocated_capacity)

# Save results
df["allocated_capacity"] = allocated
df.to_csv("results/autoscaling_buffered.csv", index=False)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["predicted_views"], label="Forecasted Views", color="orange")
plt.plot(df["date"], df["allocated_capacity"], label="Allocated Resources (with buffer)", linestyle="--", color="darkorange")
plt.axhline(100_000_000, color="red", linestyle=":", label="Scale-Up Threshold (100M)")
plt.axhline(60_000_000, color="green", linestyle=":", label="Scale-Down Threshold (60M)")
plt.title("📊 Auto-Scaling with 1-Unit Buffer (LSTM_Univariate)")
plt.xlabel("Date")
plt.ylabel("Views / Capacity")
plt.legend()
plt.tight_layout()
plt.savefig("figures/Figure_Autoscaling_Buffered_LSTM_Univariate.png")
plt.show()
