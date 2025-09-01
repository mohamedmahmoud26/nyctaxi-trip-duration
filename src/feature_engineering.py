import os
import pandas as pd
import numpy as np
import math
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from geopy.distance import geodesic  

# ------------------------
#  Data Cleaning Utilities
# ------------------------
def filter_outliers(df, column, method='zscore', threshold=3):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    if method == 'zscore':
        mask = (np.abs(zscore(df[column])) < threshold)
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[column] >= (Q1 - threshold*IQR)) & (df[column] <= (Q3 + threshold*IQR))
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'")
    
    return df[mask].copy()

def remove_rare_categories(df, categorical_cols, min_count=5):
    for col in categorical_cols:
        counts = df[col].value_counts()
        rare_vals = counts[counts < min_count].index
        df = df[~df[col].isin(rare_vals)]
    return df

# ------------------------
#  Feature Engineering Functions
# ------------------------
def add_temporal_features(df, datetime_col='pickup_datetime'):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['month'] = df[datetime_col].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['week_of_year'] = df[datetime_col].dt.isocalendar().week
    df['quarter'] = df[datetime_col].dt.quarter
    return df

# Cyclical encoding للـ time features
def add_cyclical_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1)/12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1)/12)
    
    return df

def compute_geodesic_distance(row):
    start = (row['pickup_latitude'], row['pickup_longitude'])
    end = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(start, end).km

def compute_bearing(row):
    lat1, lon1 = math.radians(row['pickup_latitude']), math.radians(row['pickup_longitude'])
    lat2, lon2 = math.radians(row['dropoff_latitude']), math.radians(row['dropoff_longitude'])
    dlon = lon2 - lon1
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    y = math.sin(dlon)*math.cos(lat2)
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

def compute_manhattan_distance(row):
    lat_diff = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111
    lon_diff = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))
    return lat_diff + lon_diff

def feature_engineer(df):
    df = add_temporal_features(df)
    df = add_cyclical_features(df)  
    df['distance_km'] = df.apply(compute_geodesic_distance, axis=1)
    df['distance_log'] = np.log1p(df['distance_km'])
    df['bearing'] = df.apply(compute_bearing, axis=1)
    df['distance_manhattan'] = df.apply(compute_manhattan_distance, axis=1)
    return df

# ------------------------
#  Process Data From CSV
# ------------------------
def process_data(train_file, val_file, test_file, out_dir, min_cat_count=5):
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    process_data_from_df(train, val, test, out_dir, min_cat_count)

# ------------------------
#  Process Data From DataFrame
# ------------------------
def process_data_from_df(train, val, test, out_dir, min_cat_count=5):
    # Log-transform target (train & val only)
    train['trip_duration'] = np.log1p(train['trip_duration'])
    val['trip_duration'] = np.log1p(val['trip_duration'])
    
    # Feature Engineering
    train = feature_engineer(train)
    val = feature_engineer(val)
    test = feature_engineer(test)

    # Remove rare categories
    categorical_cols = ['vendor_id', 'passenger_count', 'store_and_fwd_flag']
    train = remove_rare_categories(train, categorical_cols, min_cat_count)
    val = remove_rare_categories(val, categorical_cols, min_cat_count)
    test = remove_rare_categories(test, categorical_cols, min_cat_count)

    # Remove ID and datetime
    train.drop(columns=['id', 'pickup_datetime'], inplace=True, errors='ignore')
    val.drop(columns=['id', 'pickup_datetime'], inplace=True, errors='ignore')
    test.drop(columns=['id', 'pickup_datetime'], inplace=True, errors='ignore')

    # Reset indices
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # Save processed data
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)

    # Save metadata
    metadata = {
        'num_rows_train': len(train),
        'num_rows_val': len(val),
        'num_rows_test': len(test),
        'features': train.columns.tolist(),
        'categorical_features': categorical_cols,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(f"{out_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Processed data saved to {out_dir}")

# ------------------------
#  Main Execution
# ------------------------
if __name__ == "__main__":
    # Read datasets
    train = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Downloads/train (1).csv')
    val = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Downloads/val (1).csv')
    test = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Downloads/test (1).csv')  # Add test file path

    # Process data
    process_data_from_df(
        train, 
        val,
        test,
        out_dir='/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed'
    )
    print("Data processing complete.")
