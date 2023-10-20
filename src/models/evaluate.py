import json
import pickle

from pathlib import Path

import mlflow
from src import METRICS_DIR, PROCESSED_DATA_DIR,MODELS_DIR
from os import listdir
from os.path import isfile, join
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")

def load_validation_data(input_folder_path: Path):
    """Load the validation data from the prepared data folder.

    Args:
        input_folder_path (Path): Path to the prepared data folder.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the validation features and target.
    """
    images_path=input_folder_path / "valid/images"
    labels_path=input_folder_path / "valid/labels"
    X_valid = [join(images_path /f) for f in listdir(images_path) if isfile(join(images_path, f))]
    y_valid=get_validation_labels(labels_path)
    
    return X_valid, y_valid

def get_validation_labels(labels_path: Path):
    y_valid=None
    if labels_path:
        y_path = [join(labels_path /f) for f in listdir(labels_path) if isfile(join(labels_path, f))]
        y_valid={}
        for path in y_path:
            f=open(path, 'r')
            lines=[[float(value) for value in line.strip().split(" ")] for line in f.readlines()]
            y_valid[path]=lines
    return y_valid


def evaluate_model(model_file_name, x, y):
    """Evaluate the model using the validation data.

    Args:
        model_file_name (str): Filename of the model to be evaluated.
        x (pd.DataFrame): Validation features.
        y (pd.DataFrame): Validation target.

    Returns:
        Tuple[float, float]: Tuple containing the MAE and MSE values.
    """

    with open(MODELS_FOLDER_PATH / model_file_name, "rb") as pickled_model:
        yolo_model = pickle.load(pickled_model)

    
    # TO DO 
    val_predictions = yolo_model.predict(x)
    preds = [dict(boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),scores=tensor([0.536]),labels=tensor([0]),)]
    target = [dict(boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),labels=tensor([0]),)]
    metric = MeanAveragePrecision(iou_type="bbox",box_format='xywh')
    metric.update(preds, target)
    
    map=metric.compute()['map'] 
    mar_10= metric.compute()['mar_10']
    return float(map),float(mar_10)


if __name__ == "__main__":
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = METRICS_DIR

    X_valid, y_valid = load_validation_data(PROCESSED_DATA_DIR)

    mlflow.set_experiment("evaluate-model")

    with mlflow.start_run():
        # Load the model
        map, mar_10 = evaluate_model(
            "yolov8_model.pkl", X_valid, y_valid
        )

        # Save the evaluation metrics to a dictionary to be reused later
        metrics_dict = {"map": map,"mar_10":mar_10}

        # Log the evaluation metrics to MLflow
        mlflow.log_metrics(metrics_dict)

        # Save the evaluation metrics to a JSON file
        with open(metrics_folder_path / "scores.json", "w") as scores_file:
            json.dump(
                metrics_dict,
                scores_file,
                indent=4,
            )

        print("Evaluation completed.")