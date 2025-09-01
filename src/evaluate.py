from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the model on test/validation set.
    """
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"R2": r2, "RMSE": rmse}
