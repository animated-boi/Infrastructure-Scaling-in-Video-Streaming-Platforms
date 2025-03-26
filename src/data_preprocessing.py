import pandas as pd
import json


def load_and_clean_data(video_csv_path, category_json_path):
    # Load raw video data
    df = pd.read_csv(video_csv_path)

    # Load category ID mapping from JSON
    with open(category_json_path, 'r') as f:
        category_data = json.load(f)

    # Create mapping: category_id -> category_name
    category_mapping = {
        int(item['id']): item['snippet']['title']
        for item in category_data['items']
        if item['snippet']['assignable']
    }

    # Map category names to DataFrame
    df['category_name'] = df['category_id'].map(category_mapping)

    # Convert trending_date to datetime format
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')

    # Keep only useful columns
    df_cleaned = df[[
        'video_id', 'title', 'channel_title', 'category_name',
        'publish_time', 'trending_date', 'views', 'likes', 'dislikes', 'comment_count'
    ]]

    return df_cleaned


def save_cleaned_data(df_cleaned, output_path):
    df_cleaned.to_csv(output_path, index=False)


if __name__ == '__main__':
    # File paths
    VIDEO_CSV = 'data/USvideos.csv'
    CATEGORY_JSON = 'data/US_category_id.json'
    OUTPUT_CSV = 'results/cleaned_us_videos.csv'

    # Run cleaning pipeline
    cleaned_df = load_and_clean_data(VIDEO_CSV, CATEGORY_JSON)
    save_cleaned_data(cleaned_df, OUTPUT_CSV)
    print(f"Cleaned data saved to {OUTPUT_CSV}")
