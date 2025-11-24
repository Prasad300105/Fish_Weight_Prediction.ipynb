"""
Evaluate a saved pipeline on a dataset and produce simple plots.
"""
import argparse
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_pipeline(path: str):
    return joblib.load(path)


def evaluate_pipeline(pipeline_path: str, data_path: str, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    pipeline = load_pipeline(pipeline_path)
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    df = pd.read_csv(data_path)
    X = df[["Length1", "Length2", "Length3", "Height", "Width", "Species"]]
    y = df["Weight"].values

    X_trans = preprocessor.transform(X)
    preds = model.predict(X_trans)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    # save predictions
    out_df = X.copy()
    out_df["Weight_true"] = y
    out_df["Weight_pred"] = preds
    out_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # Plot: True vs Predicted
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y, y=preds)
    plt.xlabel("True Weight")
    plt.ylabel("Predicted Weight")
    plt.title("True vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_pred.png"))
    plt.close()

    # Residuals
    residuals = y - preds
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals.png"))
    plt.close()

    # Save metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    print("Saved evaluation results to", save_dir)
    print("Metrics:", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    evaluate_pipeline(args.pipeline, args.data, args.out)
