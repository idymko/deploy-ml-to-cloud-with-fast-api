import pytest
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from src.model import load_model_package, inference, compute_model_metrics
from src.data import process_data

def test_load_model_package():
    """
    Test load_model_package function
    """
    
    model, encoder, lb, scaler = load_model_package('model/model_package.pkl')
    
    assert model is not None
    assert encoder is not None
    assert scaler is not None
    assert lb is not None

@pytest.fixture
def load_data():
    model, encoder, lb, scaler = load_model_package('model/model_package.pkl')
    test = pd.read_csv("data/eval_data.csv")
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_test, y_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, 
        encoder=encoder, lb=lb, scaler=scaler
    )
    return X_test, y_test, model
    
def test_inference(load_data):
    """
    Test inference function
    """
    
    X_test, y_test, model = load_data
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
    fbeta, precision, recall, f1 = compute_model_metrics(y_true, preds)

    # Expected values based on known calculations
    expected_fbeta = 0.75
    expected_precision = 0.6
    expected_recall = 0.6667
    expected_f1 = 0.6667

    # Assertions to check if the calculated metrics match the expected values
    assert round(fbeta, 4) == expected_fbeta, f"Expected F-beta: {expected_fbeta}, but got: {fbeta}"
    assert round(precision, 4) == expected_precision, f"Expected Precision: {expected_precision}, but got: {precision}"
    assert round(recall, 4) == expected_recall, f"Expected Recall: {expected_recall}, but got: {recall}"
    assert round(f1, 4) == expected_f1, f"Expected F1: {expected_f1}, but got: {f1}"