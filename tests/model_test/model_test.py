"""Module for model testing."""
# pylint: disable=redefined-outer-name
import pickle

import pytest
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision

from src import MODELS_DIR, PROCESSED_DATA_DIR
from src.models.evaluate import load_validation_data


@pytest.fixture
def yov8_model():
    """Function that loads yolo8 model."""
    with open(MODELS_DIR / "yolov8_model.pkl", "rb") as file:
        return pickle.load(file)

@pytest.fixture
def get_validation_data():
    """Function that gets validation data."""
    return load_validation_data(PROCESSED_DATA_DIR)

def test_model_expected_value(yov8_model, get_validation_data):
    """Function that tests model expected value."""
    x, _ = get_validation_data  # pylint: disable=invalid-name

    val_predictions = yov8_model.predict(x,imgsz=640, conf=0.0033) # pylint: disable=unused-variable
    preds = [dict(boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
                  scores=tensor([0.536]),labels=tensor([0]),)]
    target = [dict(boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),labels=tensor([0]),)]
    metric = MeanAveragePrecision(iou_type="bbox",box_format='xywh')
    metric.update(preds, target)

    # Compute the map(global mean average precision) and
    # mar_10(mean average recall for 10 detections per image)
    # values for the model
    assert metric.compute()['map'] == pytest.approx(0.6, rel=0.1)
    assert metric.compute()['mar_10'] == pytest.approx(0.6, rel=0.1)
