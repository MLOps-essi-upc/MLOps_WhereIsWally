"""Main script: it includes our API initialization and endpoints."""

import base64
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List
from fastapi import FastAPI, HTTPException, Request, File, UploadFile,Response
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2
from src import MODELS_DIR, API_DIR
from ultralytics import YOLO
import os
from prometheus_fastapi_instrumentator import Instrumentator, metrics
model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Where is Wally",
    description="lorem ipsum",
    version="0.1",
)

Instrumentator().instrument(app).expose(app) #Prometheus metric tracking


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():
    """Loads all pickled models found in `MODELS_DIR` and adds them to `models_list`"""

    model_paths = [
        filename
        for filename in MODELS_DIR.iterdir()
        if filename.suffix == ".pt" and filename.stem.startswith("best")
    ]

    for path in model_paths:
        with open(path, "rb") as file:
            # model_wrapper = pickle.load(file)
            # model_wrappers_list.append(model_wrapper)
            print("file",str(file))
            model_wrapper=dict()
            model = YOLO(path)
            model_wrapper["model"]=model
            model_wrapper["type"]=str(file).split("_")[-1].split(".")[0]
            model_wrapper["info"]=model.info()
            model_wrappers_list.append(model_wrapper)


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Hello world"},
    }
    return response
    

@app.get("/models", tags=["Prediction"])
@construct_response
def _get_models_list(request: Request, type: str = None):
    """Return the list of available models"""

    available_models = [
        {
            "type": model["type"],
            "info":model["info"],
            # "parameters": model["params"],
            # "accuracy": model["metrics"],
        }
        for model in model_wrappers_list
        if model["type"] == type or type is None
    ]

    if not available_models:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Type not found")
    else:
        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": available_models,
        }



@construct_response
@app.post("/predict/{type}")
async def _predict(type: str,file: UploadFile = File(...)):
    model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    if model_wrapper:
        model=model_wrapper['model']
        contents = file.file.read()
        # contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model.predict(source=img, conf=0.25)
        boxes=results[0].boxes.xyxy
        conf=results[0].boxes.conf
        
        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
          
        return_img = annotator.result()  

        # line that fixed it
        _, encoded_img = cv2.imencode('.PNG', return_img)
        encoded_img = base64.b64encode(encoded_img)
        
    else:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Model not found"
        )
    return{
        'boxes':boxes,
        'conf':conf,
        'encoded_img': encoded_img,
    }
