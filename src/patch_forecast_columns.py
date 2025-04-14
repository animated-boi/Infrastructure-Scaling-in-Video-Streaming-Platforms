import pandas as pd
import os

def patch_forecast_file(file_path, original_column, new_column='predicted_views'):
    """
    Rename a column in a forecast CSV file to ensure it matches expected evaluation schema.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)

        if new_column in df.columns:
            print(f"✅ {file_path} already contains '{new_column}'. No patch needed.")
            return

        if original_column not in df.columns:
            print(f"⚠️ Column '{original_column}' not found in {file_path}. Columns available: {list(df.columns)}")
            return

        df.rename(columns={original_column: new_column}, inplace=True)
        df.to_csv(file_path, index=False)
        print(f"✅ Patched {file_path}: Renamed '{original_column}' → '{new_column}'")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

if __name__ == "__main__":
    print("🔧 Patching forecast files for Prophet and ARIMA compatibility...\n")
    patch_forecast_file("results/prophet_forecast.csv", original_column="yhat")
    patch_forecast_file("results/arima_forecast.csv", original_column="forecast_views")
    print("\n✅ Done.\n")
