import pandas as pd
import matplotlib.pyplot as plt
import os

# Load LSTM forecast with actuals
df = pd.read_csv("results/lstm_final_forecast.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Forecasted and actual views
views = df["predicted_views"].values
actual_views = df["actual_views"].values

# Parameters
unit_capacity = 50_000_000      # 1 unit handles 50M views
buffer_units = 0.5              # Max buffer: 0.5 extra units
max_scaling_cap = 1_000_000_000 # Optional max cap

allocated = []
alerts = []

for idx, v in enumerate(views):
    required_units = int((v + unit_capacity - 1) // unit_capacity)
    allocated_units = required_units + buffer_units
    allocated_capacity = min(allocated_units * unit_capacity, max_scaling_cap)
    allocated.append(allocated_capacity)

    # Alert logic
    act = actual_views[idx]
    if act > allocated_capacity:
        alerts.append(f"{df['date'].iloc[idx].date()} ⚠️ Scale-Up Needed: Actual={act:.0f}, Allocated={allocated_capacity:.0f}")
    elif act < 0.5 * allocated_capacity:
        alerts.append(f"{df['date'].iloc[idx].date()} 📉 Scale-Down Opportunity: Actual={act:.0f}, Allocated={allocated_capacity:.0f}")

# Log alerts
print("🔔 Alerts:")
for a in alerts:
    print(a)

# Save with new allocated column
df["allocated_capacity"] = allocated
df.to_csv("results/autoscaling_buffered_alerts.csv", index=False)

# 📈 Plot
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["actual_views"], label="Actual Views", color="blue", alpha=0.6)
plt.plot(df["date"], df["predicted_views"], label="Forecasted Views", color="orange", alpha=0.8)
plt.plot(df["date"], df["allocated_capacity"], label="Allocated Resources (buffered)", linestyle="--", color="darkorange")
plt.axhline(100_000_000, color="red", linestyle=":", label="Scale-Up Threshold (100M)")
plt.axhline(60_000_000, color="green", linestyle=":", label="Scale-Down Threshold (60M)")
plt.title("📊 Auto-Scaling with 0.5-Unit Buffer and Alerting")
plt.xlabel("Date")
plt.ylabel("Views / Capacity")
plt.legend()
plt.tight_layout()
plt.savefig("figures/Figure_Autoscaling_Buffered_Alerting.png")
plt.show()
