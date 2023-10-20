import pickle
import pytest
from src import MODELS_DIR, PROCESSED_DATA_DIR,RAW_DATA_DIR
from src.models.evaluate import load_validation_data,get_validation_labels
from src.features.process_data import noise_removal
from torch import tensor
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

@pytest.fixture
def validation_data_path():
    return PROCESSED_DATA_DIR
 
@pytest.fixture
def labels_path():
    return PROCESSED_DATA_DIR / "valid/labels"

@pytest.fixture
def raw_images_path():
    images_path=RAW_DATA_DIR / "train/images"
    data_path = [join(images_path /f) for f in listdir(images_path) if isfile(join(images_path, f))]
    return data_path

#validation_labels function's tests
def test_validation_labels_ObjnotNone(labels_path):
     assert get_validation_labels(labels_path) is not None
     

def test_validation_labels_isRightDatatype(labels_path):
     assert isinstance(get_validation_labels(labels_path), dict)
     

def test_validation_labels_ObjIsNone(labels_path):
     assert get_validation_labels(None) is None

     
def test_validation_labels_path_fails(labels_path):
    with pytest.raises(FileNotFoundError):
        assert isinstance(get_validation_labels(labels_path / "poo"), dict)
    

#load_validation_data function's tests
def test_validation_data_isAllObjectsReturned(validation_data_path):
    assert len(load_validation_data(validation_data_path))==2
    
def test_validation_data_isRightDatatype(validation_data_path):
    assert isinstance(load_validation_data(validation_data_path)[0], list)
   