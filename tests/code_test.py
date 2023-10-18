import pickle
import pytest
from src import MODELS_DIR, PROCESSED_DATA_DIR
from src.models.evaluate import load_validation_data,get_validation_labels
from torch import tensor

@pytest.fixture
def labels_path():
    return PROCESSED_DATA_DIR / "valid/labels"

@pytest.fixture
def validation_data_path():
    return PROCESSED_DATA_DIR 

def test_validation_labels(labels_path):
    assert get_validation_labels(labels_path) is not None
    assert isinstance(get_validation_labels(labels_path), dict)
    assert get_validation_labels(None) is None
    with pytest.raises(FileNotFoundError):
        assert isinstance(get_validation_labels(labels_path / "poo"), dict)
    

def test_validation_data(validation_data_path):
    assert len(load_validation_data(validation_data_path))==2
    assert isinstance(load_validation_data(validation_data_path)[0], list)
   
    
