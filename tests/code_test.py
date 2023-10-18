import pickle
import pytest
from src import MODELS_DIR, PROCESSED_DATA_DIR,RAW_DATA_DIR
from src.models.evaluate import load_validation_data,get_validation_labels
from src.models.process_data import noise_removal
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

def test_validation_labels(labels_path):
    assert get_validation_labels(labels_path) is not None
    assert isinstance(get_validation_labels(labels_path), dict)
    assert get_validation_labels(None) is None
    with pytest.raises(FileNotFoundError):
        assert isinstance(get_validation_labels(labels_path / "poo"), dict)
    

def test_validation_data(validation_data_path):
    assert len(load_validation_data(validation_data_path))==2
    assert isinstance(load_validation_data(validation_data_path)[0], list)
   
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def test_noise_removal(raw_images_path):
    img = cv2.imread(raw_images_path[0]) 
    denoised=noise_removal(raw_images_path[0])
    assert denoised is not None
    assert (img.shape == denoised.shape)
    assert (np.bitwise_xor(img,denoised).any())
    