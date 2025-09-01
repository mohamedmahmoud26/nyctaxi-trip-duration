from sklearn.model_selection import cross_val_score
import numpy as np

def run_cross_validation(pipeline, X, y, cv=5):
    """
    Run cross-validation and return mean R2 score.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    return np.mean(scores), scores
