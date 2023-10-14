import mlflow
import os
from getpass import getpass
from ultralytics import YOLO


os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
os.environ['MLFLOW_TRACKING_URI'] = input('Enter your DAGsHub project tracking URI: ')

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Load the model.
model = YOLO('yolov8n.pt')

# Training.
mlflow.set_experiment("yolo_v8_model_training")
with mlflow.start_run(run_name="yolo_v8_model_training"):
  results = model.train(
    data='../../data/yolov8_format/data.yaml',
    imgsz=640,
    epochs=30,
    batch=8,
    name='yolov8n_custom')

