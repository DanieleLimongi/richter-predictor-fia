import argparse
import pandas as pd
import joblib
from src.data_loading import load_train, load_test
from src.preprocessing import build_preprocessor
from src.model_training import train_model
from src.evaluation import macro_f1

def main(mode):
    print(f">>> Modalità selezionata: {mode}")
    if mode == "train":
        X, y = load_train()
        preprocessor = build_preprocessor(X)
        train_model(X, y, preprocessor)
    elif mode == "predict":
        X_test = load_test()
        clf = joblib.load("models/mlp_model.pkl")
        predictions = clf.predict(X_test)
        pd.DataFrame({
            "building_id": X_test["building_id"],
            "damage_grade": predictions
        }).to_csv("submission.csv", index=False)
    else:
        print("Modalità non riconosciuta. Usa --mode train oppure --mode predict.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    main(args.mode)
