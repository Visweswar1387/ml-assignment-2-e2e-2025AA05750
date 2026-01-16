from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def xg_boost(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model