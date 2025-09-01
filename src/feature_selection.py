import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Load processed data
# ------------------------
train = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/train.csv')
val   = pd.read_csv('/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/val.csv')

# Split features and target
X_train = train.drop(columns=['trip_duration'])
y_train = train['trip_duration']
X_val   = val.drop(columns=['trip_duration'])
y_val   = val['trip_duration']

# ------------------------
# Encode categorical column
# ------------------------
X_train['store_and_fwd_flag'] = X_train['store_and_fwd_flag'].map({'N':0, 'Y':1})
X_val['store_and_fwd_flag']   = X_val['store_and_fwd_flag'].map({'N':0, 'Y':1})

# ------------------------
# Train XGBoost Regressor
# ------------------------
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ------------------------
# SHAP Explainer
# ------------------------
explainer = shap.Explainer(model)
shap_values = explainer(X_val)

# ------------------------
# Global Feature Importance
# ------------------------
# Bar plot
shap.summary_plot(shap_values, X_val, plot_type="bar")

# Detailed summary plot
shap.summary_plot(shap_values, X_val)

# ------------------------
# Extract SHAP Feature Importance as DataFrame
# ------------------------
shap_importance = pd.DataFrame({
    'feature': X_val.columns,
    'shap_importance': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='shap_importance', ascending=False)

# Show top features
print(shap_importance)

# Save to CSV
shap_importance.to_csv(
    '/mnt/c/Users/Mohamed Mahmoud/Trip_Duration/nyc-taxi-trip-duration/data_processed/shap_feature_importance.csv',
    index=False
)

# ------------------------
# Optional: Top 10 Features
# ------------------------
top_10_features = shap_importance.head(10)
print("\nTop 10 Features:\n", top_10_features)
