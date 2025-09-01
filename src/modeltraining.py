def train_model(pipeline, X_train, y_train):
    """
    Train pipeline on training data.
    """
    pipeline.fit(X_train, y_train)
    return pipeline
