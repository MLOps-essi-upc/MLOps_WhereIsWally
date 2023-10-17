import mlflow
import os
from getpass import getpass
from ultralytics import YOLO
import yaml


os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
os.environ['MLFLOW_TRACKING_URI'] = input('Enter your DAGsHub project tracking URI: ')

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

with open(r"params.yaml") as f:
    params = yaml.safe_load(f)

# Load the model.
model = YOLO(params['model_type'])

# Training.
mlflow.set_experiment(params['name'])
with mlflow.start_run(run_name=params['name']):
  results = model.train(
    data='../../data/yolov8_format/data.yaml',
    imgsz=params['imgsz'],
    epochs=params['epochs'],
    batch=params['batch'],
    name=params['name'])

