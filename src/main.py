import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

# ==============================
# Functions
# ==============================
def build_pipeline(numeric_features, model_type="ridge", alpha=1.0):
    """
    Create a pipeline with preprocessing + model.
    Only numeric features here.
    """
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features)
    ])

    if model_type == "ridge":
        model = Ridge(alpha=alpha, random_state=42)
    else:
        raise ValueError("Only 'ridge' is supported")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def run_cross_validation(pipeline, X, y, cv=5, scoring="r2"):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    return np.mean(scores), scores

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return {"R2": r2, "RMSE": rmse, "RMSLE": rmsle}, y_pred

# ==============================
# Load processed data
# ==============================
train_path = "/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/train.csv"
val_path = "/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/val.csv"
test_path = "/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/test.csv"

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)

# ==============================
# Select features
# ==============================
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

X_train = train[selected_features]
y_train = train["trip_duration"]

X_val = val[selected_features]
y_val = val["trip_duration"]

X_test = test[selected_features]

# ==============================
# Build Pipeline
# ==============================
pipe = build_pipeline(numeric_features=selected_features, model_type="ridge", alpha=1.0)

# ==============================
# Cross-validation
# ==============================
mean_r2, r2_scores = run_cross_validation(pipe, X_train, y_train, cv=5)
print(f"Cross-Validation Mean R2: {mean_r2:.4f}")
print(f"R2 Scores per fold: {r2_scores}")

# ==============================
# Train Model
# ==============================
trained_pipe = train_model(pipe, X_train, y_train)

# ==============================
# Evaluate on Validation Set
# ==============================
results, y_val_pred = evaluate_model(trained_pipe, X_val, y_val)
print(f"Validation R²: {results['R2']:.4f}")
print(f"Validation RMSE: {results['RMSE']:.4f}")
print(f"Validation RMSLE: {results['RMSLE']:.4f}")

# ==============================
# Predict on Test Set
# ==============================
test_preds = trained_pipe.predict(X_test)
test["trip_duration_pred"] = test_preds

# ==============================
# Create folders
# ==============================
os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)
os.makedirs("../submissions", exist_ok=True)

# ==============================
# Save model
# ==============================
joblib.dump(trained_pipe, "../models/ridge_pipeline.pkl")

# ==============================
# Save test predictions
# ==============================
if "id" in test.columns:
    test[["id", "trip_duration_pred"]].to_csv("../submissions/test_predictions.csv", index=False)
else:
    test[["trip_duration_pred"]].to_csv("../submissions/test_predictions.csv", index=False)

# ==============================
# Save processed datasets in results
# ==============================
train.to_csv("../results/train_processed.csv", index=False)
val.to_csv("../results/val_processed.csv", index=False)
test.to_csv("../results/test_with_predictions.csv", index=False)

# ==============================
# Save validation metrics
# ==============================
metrics_df = pd.DataFrame({
    "Metric": ["R2", "RMSE", "RMSLE"],
    "Value": [results["R2"], results["RMSE"], results["RMSLE"]]
})
metrics_df.to_csv("../results/validation_metrics.csv", index=False)

print("All results saved in ../results/")
print("Model saved in ../models/ridge_pipeline.pkl")
print("Predictions saved in ../submissions/test_predictions.csv")
