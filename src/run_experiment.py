import os
import csv
import yaml
import pandas as pd
from pathlib import Path

from data_loader import load_csv
from preprocessing import split_data, normalize_data
from model import train_sklearn, train_scratch
from evaluation import metrics


def main():
    # Always run from project root
    project_root = Path(__file__).resolve().parents[1]

    # Load config safely
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data
    csv_path = project_root / cfg["data"]["csv_path"]
    df = load_csv(csv_path)

    # Feature/target
    X = df[[cfg["experiment"]["feature_column"]]]
    y = df[cfg["experiment"]["target_column"]]

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        cfg["experiment"]["test_size"],
        cfg["project"]["random_seed"]
    )

    # Normalize
    if cfg["experiment"]["normalize"]:
        X_train, X_test = normalize_data(X_train, X_test)

    # Sklearn model
    sk_model = train_sklearn(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_metrics = metrics(y_test, sk_pred)

    # Scratch model
    w, b = train_scratch(
        X_train.flatten(), y_train.values,
        cfg["manual_lr"]["learning_rate"],
        cfg["manual_lr"]["iterations"]
    )
    scratch_pred = w * X_test.flatten() + b
    scratch_metrics = metrics(y_test, scratch_pred)

    # Save results (append)
    results_csv = project_root / cfg["outputs"]["results_csv"]
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "rmse", "mae", "r2"]
    file_exists = results_csv.exists()

    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": "sklearn", **sk_metrics})
        writer.writerow({"model": "scratch", **scratch_metrics})

    print("Experiment complete. Results saved to:", results_csv)


if __name__ == "__main__":
    main()
