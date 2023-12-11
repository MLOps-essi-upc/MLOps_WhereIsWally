# pylint: disable=redefined-outer-name
import os
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from src.app.backend.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    with TestClient(app) as client:
        return client

def test_get_main(client):
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()
    assert "message" in response_body
    assert "data" in response_body
    assert "message" in response_body["data"]
    assert response_body["message"] == HTTPStatus.OK.phrase
    assert response_body["data"]["message"] == "Welcome to Where is Wally!"


def test_predict_with_invalid_file(client):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'testing_file.txt')
    with open(file_path, 'rb') as file:
        response = client.post("/predict/all", files={"file": file})
    assert response.status_code == 400
    assert "Invalid image file" in response.text


def test_predict_with_valid_image_wally_found_model_all(client):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'testing_img.png')
    with open(file_path, 'rb') as file:
        response = client.post("/predict/all", files={"file": file})
    assert response.status_code == 200
    response_body = response.json()
    assert response_body['found'] is True
    assert "boxes" in  response_body
    assert 'orig_shape' in response_body['boxes']
    assert len(response_body['boxes']['orig_shape']) > 0


def test_predict_with_valid_image_wally_not_found(client):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'testing_not_found_img.jpeg')
    with open(file_path, 'rb') as file:
        response = client.post("/predict/all", files={"file": file})
    assert response.status_code == 200
    response_body = response.json()
    assert "boxes" in  response_body
    assert "found" in  response_body
    assert response_body['found'] is False
    assert "Processing completed, but Wally was not found in the image." in response.text
    assert len(response_body['boxes']) == 0


def test_predict_with_valid_image_model_wally(client):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'testing_img.png')
    with open(file_path, 'rb') as file:
        response = client.post("/predict/wally", files={"file": file})
    assert response.status_code == 200
    response_body = response.json()
    assert response_body['found'] is True
    assert "boxes" in  response_body
    assert 'orig_shape' in response_body['boxes']
    assert len(response_body['boxes']['orig_shape']) > 0

