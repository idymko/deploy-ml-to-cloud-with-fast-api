import pandas as pd

from src.model import load_model_package
from src.data import process_data

def test_process_data():
    
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
    
    test = pd.read_csv("data/eval_data.csv")
    X, y, encoder, lb, scaler = process_data(
        test, categorical_features=cat_features, label="salary", training=True
    )
    
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert X.shape[0] == y.shape[0]
    assert encoder is not None
    assert lb is not None
    assert scaler is not None
    