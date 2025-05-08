import pandas as pd

def load_train(path="data/train_values.csv",
               labels_path="data/train_labels.csv",
               target_col="damage_grade"):
    X = pd.read_csv(path)
    # carica soltanto la colonna di target
    labels = pd.read_csv(labels_path)
    y = labels[target_col]
    return X, y

def load_test(path="data/test_values.csv"):
    return pd.read_csv(path)
