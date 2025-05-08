import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

def train_model(X, y, preprocessor, model_path="models/mlp_model.pkl"):
    clf = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))
    ])
    clf.fit(X, y)
    joblib.dump(clf, model_path)
