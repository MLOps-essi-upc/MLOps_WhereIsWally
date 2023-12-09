# pylint: disable=redefined-outer-name
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import pytest

from src import RAW_DATA_DIR
from src.features.process_data import noise_removal


@pytest.fixture
def raw_images_path():
    images_path=RAW_DATA_DIR / "train/images"
    data_path = [join(images_path /f) for f in listdir(images_path) if isfile(join(images_path, f))]
    return data_path


def is_similar(image1, image2):
    return image1.shape == image2.shape and not np.bitwise_xor(image1,image2).any()

def test_noise_removal_is_obj_not_none(raw_images_path):
    denoised=noise_removal(raw_images_path[0])
    assert denoised is not None


def test_noise_removal_is_equal_shape(raw_images_path):
    img = cv2.imread(raw_images_path[0])
    denoised=noise_removal(raw_images_path[0])
    assert img.shape == denoised.shape

def test_noise_removal_is_not_same_image(raw_images_path):
    img = cv2.imread(raw_images_path[0])
    denoised=noise_removal(raw_images_path[0])
    assert (np.bitwise_xor(img,denoised).any())
