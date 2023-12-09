"""Main script: it includes our API initialization and endpoints."""

import asyncio
import base64
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from src import MODELS_DIR
from ultralytics import YOLO
import os
import asyncio
from prometheus_fastapi_instrumentator import Instrumentator, metrics

model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Where is Wally",
    description="Upload an image and we will help you to find Wally",
    version="0.1",
)

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=False,
        should_include_status=True,
        metric_namespace="a",
        metric_subsystem="b",
    )
).add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=False,
        should_include_status=True,
        metric_namespace="namespace",
        metric_subsystem="subsystem",
    )
)

instrumentator.instrument(app)

instrumentator.expose(app, include_in_schema=False, should_gzip=True)


def construct_response(f):
    @wraps(f)
    async def wrap(request: Request, *args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(f):
                results = await f(request, *args, **kwargs)
            else:
                results = f(request, *args, **kwargs)

            # Default status code
            status_code = results.get("status-code", HTTPStatus.OK)

            response = {
                "message": results.get("message", status_code.phrase),
                "method": request.method,
                "status-code": status_code,
                "timestamp": datetime.now().isoformat(),
                "url": request.url._url,
                "data": results.get("data", {}),
                "found": results.get("found", None),
            }

            # Include additional keys if present
            for key in ["boxes", "conf", "encoded_img"]:
                if key in results:
                    response[key] = results[key]

            return response

        except HTTPException as http_exc:
            # Forward HTTP exceptions as they are
            raise http_exc

        except Exception as exc:
            # Handle other exceptions
            return {
                "message": "An error occurred",
                "method": request.method,
                "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "timestamp": datetime.now().isoformat(),
                "url": request.url._url,
                "detail": str(exc),
            }

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
            model_wrapper = dict()
            model = YOLO(path)
            model_wrapper["model"] = model
            model_wrapper["type"] = str(file).split("_")[-1].split(".")[0]
            model_wrapper["info"] = model.info()
            model_wrappers_list.append(model_wrapper)


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Where is Wally!"},
    }
    return response


@app.get("/models", tags=["Prediction"])
@construct_response
def _get_models_list(request: Request, type: str = None):
    """Return the list of available models"""

    available_models = [
        {
            "type": model["type"],
            "info": model["info"],
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
async def _predict(type: str, file: UploadFile = File(...)):
    model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)

    if not model_wrapper:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Model not found"
        )

    else:
        model = model_wrapper["model"]
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="Invalid image file"
            )
        else:
            results = model.predict(source=img, conf=0.25)
            boxes = results[0].boxes.xyxy
            conf = results[0].boxes.conf

            for r in results:
                annotator = Annotator(img)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[
                        0
                    ]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    color = (0, 0, 0)
                    annotator.box_label(b, model.names[int(c)], color=color)

            return_img = annotator.result()

            _, encoded_img = cv2.imencode(".PNG", return_img)
            encoded_img = base64.b64encode(encoded_img)
            is_empty = len(boxes) == 0
            if is_empty:
                return {
                    "boxes": boxes,
                    "encoded_img": encoded_img.decode(),
                    "message": "Processing completed, but Wally was not found in the image.",
                    "found": False,
                }

            return {
                "boxes": boxes,
                "conf": conf,
                "encoded_img": encoded_img.decode(),
                "found": True,
            }
