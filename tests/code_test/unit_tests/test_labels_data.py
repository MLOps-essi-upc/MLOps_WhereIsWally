# pylint: disable=redefined-outer-name
from os import listdir
from os.path import isfile, join

import pytest

from src import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models.evaluate import get_validation_labels, load_validation_data


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
def test_validation_labels_obj_not_none(labels_path):
    assert get_validation_labels(labels_path) is not None


def test_validation_labels_is_right_data_type(labels_path):
    assert isinstance(get_validation_labels(labels_path), dict)


def test_validation_labels_obj_is_none():
    assert get_validation_labels(None) is None


def test_validation_labels_path_fails(labels_path):
    with pytest.raises(FileNotFoundError):
        assert isinstance(get_validation_labels(labels_path / "poo"), dict)


#load_validation_data function's tests
def test_validation_data_is_all_objects_returned(validation_data_path):
    assert len(load_validation_data(validation_data_path))==2

def test_validation_data_is_right_data_type(validation_data_path):
    assert isinstance(load_validation_data(validation_data_path)[0], list)
