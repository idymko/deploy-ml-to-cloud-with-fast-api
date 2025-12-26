import pytest
import pandas as pd
from src.model import load_model_package, inference, compute_model_metrics

def test_load_model_package():
    """
    Test load_model_package function
    """
    
    model, label_encoder = load_model_package('testing/model.pkl')
    assert model is not None
    assert label_encoder is not None

@pytest.fixture
def load_data():
    model, label_encoder = load_model_package('testing/model.pkl')
    X_test = pd.read_csv("testing/test_data.csv")
    y_test = X_test.pop("salary")
    
    return X_test, y_test, model, label_encoder
    
def test_inference(load_data):
    """
    Test inference function
    """
    
    X_test, y_test, model, label_encoder = load_data
    preds = inference(model, X_test)
    
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})
    
def test_compute_model_metrics():
    """
    Test compute_model_metrics function
    """
    
    # Sample true labels and predictions
    y_true =    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    preds =     [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]

    # Call the calculate_metrics function
    fbeta, precision, recall = compute_model_metrics(y_true, preds)

    # Expected values based on known calculations
    expected_fbeta = 0.75
    expected_precision = 0.6
    expected_recall = 0.6667

    # Assertions to check if the calculated metrics match the expected values
    assert round(fbeta, 4) == expected_fbeta, f"Expected F-beta: {expected_fbeta}, but got: {fbeta}"
    assert round(precision, 4) == expected_precision, f"Expected Precision: {expected_precision}, but got: {precision}"
    assert round(recall, 4) == expected_recall, f"Expected Recall: {expected_recall}, but got: {recall}"