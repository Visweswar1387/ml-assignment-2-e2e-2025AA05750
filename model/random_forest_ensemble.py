from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def random_forest(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model