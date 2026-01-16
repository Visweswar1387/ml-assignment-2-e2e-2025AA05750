from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

def naive_bayes(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])
    model.fit(X_train, y_train)
    return model