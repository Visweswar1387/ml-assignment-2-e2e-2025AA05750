from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

def k_nearest(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    return model