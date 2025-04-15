# simulation_cost_estimation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
import os

# Configuration
UNIT_CAPACITY = 50_000_000           # 50M views per unit
UNIT_COST = 1000                     # $1000 per unit per day
UNDERPROVISION_PENALTY = 0.00001     # $ per missed view
OVERPROVISION_PENALTY = 0.000002     # $ per unused view
BUFFER_UNITS = 0.5                   # buffer on top of forecasted demand

# Load forecast and actual data
df = pd.read_csv("results/lstm_final_forecast.csv", parse_dates=["date"])
df = df.sort_values("date")

# Initialize cost tracking
cost_data = []

for _, row in df.iterrows():
    forecast = row["predicted_views"]
    actual = row["actual_views"]
    date = row["date"]

    # Compute allocation
    raw_units = forecast / UNIT_CAPACITY
    allocated_units = int(np.ceil(raw_units + BUFFER_UNITS))
    allocated_capacity = allocated_units * UNIT_CAPACITY

    # Compute costs
    daily_unit_cost = allocated_units * UNIT_COST
    if actual > allocated_capacity:
        under_penalty = (actual - allocated_capacity) * UNDERPROVISION_PENALTY
        over_penalty = 0
    else:
        under_penalty = 0
        over_penalty = (allocated_capacity - actual) * OVERPROVISION_PENALTY

    total_cost = daily_unit_cost + under_penalty + over_penalty

    # Profit Logic
    usage_fee = UNIT_COST / UNIT_CAPACITY
    ideal_fee = usage_fee  # per view ideally
    profit = (
        (forecast - actual) * ideal_fee
        - (actual - forecast) * UNDERPROVISION_PENALTY
        - (forecast - actual) * UNDERPROVISION_PENALTY
    )

    profit_or_loss = profit
    profit_str = f"+${abs(profit_or_loss):,.2f}" if profit_or_loss >= 0 else f"-${abs(profit_or_loss):,.2f}"

    cost_data.append({
        "Date": date.strftime('%Y-%m-%d'),
        "Forecasted Views": f"{forecast:,.0f}",
        "Actual Views": f"{actual:,.0f}",
        "Allocated Units": allocated_units,
        "Unit Cost ($)": f"{daily_unit_cost:,.2f}",
        "Underprovision Penalty ($)": f"{under_penalty:,.2f}",
        "Overprovision Penalty ($)": f"{over_penalty:,.2f}",
        "Total Cost ($)": f"{total_cost:,.2f}",
        "Profit/Loss ($)": profit_str
    })

# Create DataFrame
cost_df = pd.DataFrame(cost_data)

# Save CSV
cost_df.to_csv("results/simulation_cost_analysis.csv", index=False)

# Plot styled table as figure
fig, ax = plt.subplots(figsize=(20, len(cost_df) * 0.5))
ax.axis("off")
tbl = table(ax, cost_df, loc="center", cellLoc='center', colWidths=[0.1]*len(cost_df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)

plt.title("📊 Autoscaling Cost Summary with Profit/Loss", fontsize=16, pad=20)
plt.savefig("figures/Figure_Autoscaling_Cost_Table.png", bbox_inches="tight", dpi=300)
plt.show()

print("✅ Cost summary with Profit/Loss saved as image and CSV.")
