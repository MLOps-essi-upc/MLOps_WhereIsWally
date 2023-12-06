"""Module that trains the model """
import glob
import os
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

from src import (
    ARTIFACTS_DIR,
    DATA_YAML_DIR,
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT_DIR,
)

os.environ["MLFLOW_TRACKING_USERNAME"] = input("Enter your DAGsHub username: ")
os.environ["MLFLOW_TRACKING_PASSWORD"] = input("Enter your DAGsHub access token: ")
os.environ[
    "MLFLOW_TRACKING_URI"
] = "https://dagshub.com/Sebastianpaglia/MLOps_WhereIsWally.mlflow"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

EMISSIONS_OUTPUT_FOLDER = METRICS_DIR

with open(ROOT_DIR / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Load the model.
model = YOLO(params["model_type"])

# Training.
mlflow.set_experiment(params["name"])
with mlflow.start_run():
    with EmissionsTracker(
        output_dir=EMISSIONS_OUTPUT_FOLDER,
        output_file="emissions.csv",
        on_csv_write="update",
    ):
        results = (
            model.train(
                data=DATA_YAML_DIR,
                imgsz=params["imgsz"],
                epochs=params["epochs"],
                batch=params["batch"],
                name=params["name"],
            ),
        )

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(EMISSIONS_OUTPUT_FOLDER + "/emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # Save the model as a pickle file
    Path("models").mkdir(exist_ok=True)

    last_run_path = max(
        glob.glob(os.path.join(ARTIFACTS_DIR, "*/")), key=os.path.getmtime
    )
    best_weight_path = ARTIFACTS_DIR / last_run_path / "weights/best.pt"
    train_params_file = ARTIFACTS_DIR / last_run_path / "args.yaml"
    train_metrics_file = ARTIFACTS_DIR / last_run_path / "results.csv"
    shutil.copy(best_weight_path, MODELS_DIR / "model.pt")
    shutil.copy(train_params_file, REPORTS_DIR / "train_params.yaml")
    shutil.copy(train_params_file, REPORTS_DIR / "train_metrics.csv")

    # with open(MODELS_DIR / "yolov8_model.pkl", "wb") as pickle_file:
    #     pickle.dump(results, pickle_file)
