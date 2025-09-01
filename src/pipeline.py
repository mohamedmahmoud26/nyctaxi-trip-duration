from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

def build_pipeline(numeric_features, categorical_features, alpha=1.0):
    """
    Create a pipeline with OneHotEncoder for categorical, 
    StandardScaler for numeric, and Ridge Regression.
    """
    # Preprocessor: scale numeric, encode categorical
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    
    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("ridge", Ridge(alpha=alpha, random_state=42))
    ])
    
    return pipeline
