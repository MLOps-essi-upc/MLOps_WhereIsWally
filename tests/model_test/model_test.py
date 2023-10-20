import pickle
import pytest
from src import MODELS_DIR, PROCESSED_DATA_DIR
from src.models.evaluate import load_validation_data
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
import torch
from map_boxes import mean_average_precision_for_boxes

@pytest.fixture
def yov8_model():
    with open(MODELS_DIR / "yolov8_model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture
def get_validation_data():
    return load_validation_data(PROCESSED_DATA_DIR)


def test_model_expected_value(yov8_model, get_validation_data):
    x, y = get_validation_data

    val_predictions = yov8_model.predict(x,imgsz=640, conf=0.0033)
    predictions_lst=[]
    labels_lst=[]
   
    # for img,prediction in zip(x,val_predictions):
    #     img_name=img.split("/")[0]
    #     masks = prediction.masks  # Masks object for segmentation masks outputs
    #     probs = prediction.probs  # Class probabilities for classification outputs
    #     if probs:
    #         label=int(max(probs))
    #     else:
    #         label=1
    #     print(y)
    #     predictions_lst.append([img_name,label]+[v for v in prediction.boxes.xywh])
    #     print(y[img])
    #     labels_lst.append([img_name],int(y[img][0]),0.1,y[img][1],y[img][2],y[img][3],y[img][4])
   
    # mean_ap, average_precisions = mean_average_precision_for_boxes(labels_lst, predictions_lst)
  
    # preds = [dict(boxes=torch.FloatTensor(next(iter(y.values()))[0]),scores=tensor([0.536]),labels=tensor([[1],[2],[2]]),)]
    # target = [dict(boxes=boxes.xywh,scores=tensor([0.536]),labels=tensor([[1],[1],[1]]),)]
    preds = [dict(boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),scores=tensor([0.536]),labels=tensor([0]),)]
    target = [dict(boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),labels=tensor([0]),)]
    metric = MeanAveragePrecision(iou_type="bbox",box_format='xywh')
    metric.update(preds, target)
    
    # Compute the map(global mean average precision) and mar_10(mean average recall for 10 detections per image)
    # values for the model
    assert metric.compute()['map'] == pytest.approx(0.6, rel=0.1)
    assert metric.compute()['mar_10'] == pytest.approx(0.6, rel=0.1)
    
  