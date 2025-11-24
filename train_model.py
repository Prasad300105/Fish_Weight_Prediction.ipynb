"""
Train and save a regression model (RandomForest) for fish weight prediction.
"""
import argparse
import joblib
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_preprocessing import load_data, preprocess_df, build_preprocessor


def train(data_path: str, output_path: str, random_state: int = 42, test_size: float = 0.2):
    df = load_data(data_path)

    # Preprocess and get fitted preprocessor
    X, y, preprocessor = preprocess_df(df, preprocessor=None)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }

    # Save pipeline (preprocessor + model)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pipeline = {"preprocessor": preprocessor, "model": model}
    joblib.dump(pipeline, output_path)

    # Save metrics
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved pipeline to {output_path}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to Fish.csv")
    parser.add_argument("--output", type=str, default="models/fish_model.joblib", help="output joblib path")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    train(args.data_path, args.output, args.random_state)
