import pandas as pd

def aggregate_by_date_and_category(cleaned_csv_path, output_path):
    # Load the cleaned data
    df = pd.read_csv(cleaned_csv_path)

    # Group by date and category, then aggregate
    agg_df = df.groupby(['trending_date', 'category_name']).agg({
        'views': 'sum',
        'likes': 'mean',
        'video_id': 'count'  # number of trending videos
    }).reset_index()

    # Rename columns for clarity
    agg_df.rename(columns={
        'likes': 'avg_likes',
        'video_id': 'num_trending_videos'
    }, inplace=True)

    # Save to CSV
    agg_df.to_csv(output_path, index=False)
    print(f"Aggregated time series saved to {output_path}")


if __name__ == '__main__':
    INPUT_CSV = 'results/cleaned_us_videos.csv'
    OUTPUT_CSV = 'results/aggregated_timeseries.csv'

    aggregate_by_date_and_category(INPUT_CSV, OUTPUT_CSV)
