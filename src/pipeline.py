import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

# ------------------------
# Selected Features
# ------------------------
selected_features = [
    "distance_km",
    "distance_log",
    "distance_manhattan",
    "bearing",
    "pickup_latitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "is_weekend", "week_of_year", "quarter"
]

# ------------------------
# Build Pipeline Function
# ------------------------
def build_pipeline(features, alpha=1.0):
    """
    Pipeline for numeric features with Ridge Regression.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),   # Scale numeric features
        ("ridge", Ridge(alpha=alpha, random_state=42))
    ])
    return pipeline

# ------------------------
# Load Data
# ------------------------
train = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/train.csv')
val   = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/val.csv')

# Use only selected features
X_train = train[selected_features]
y_train = train['trip_duration']
X_val   = val[selected_features]
y_val   = val['trip_duration']

# ------------------------
# Build and Train Pipeline
# ------------------------
pipeline = build_pipeline(selected_features, alpha=1.0)
pipeline.fit(X_train, y_train)

# ------------------------
# Predict on Validation
# ------------------------
y_pred = pipeline.predict(X_val)

# ------------------------
# Save Predictions as CSV
# ------------------------
os.makedirs("../results", exist_ok=True)
predictions_df = pd.DataFrame({
    'actual': y_val,
    'predicted': y_pred
})
predictions_path = "../results/predictions.csv"
predictions_df.to_csv(predictions_path, index=False)
print(f" Predictions saved to: {predictions_path}")

# ------------------------
# Save Selected Features as CSV
# ------------------------
selected_df = pd.DataFrame({'Selected_Features': selected_features})
selected_path = "../results/selected_features.csv"
selected_df.to_csv(selected_path, index=False)
print(f" Selected features saved to: {selected_path}")
