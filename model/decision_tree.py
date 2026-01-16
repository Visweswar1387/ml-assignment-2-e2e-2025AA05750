from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def decision_tree(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)
    return model