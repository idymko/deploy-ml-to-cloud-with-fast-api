# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd 
import joblib
import json 

import yaml

# Add the necessary imports for the starter code.
from src.data import process_data
from src.model import train_model, compute_model_metrics, inference

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load in the data.    
data_path = params['train']['data_path']
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

## Dump some data for testing. 
test.to_csv('data/eval_data.csv', index=False)

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
X_train, y_train, encoder, lb, scaler = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train model
model = train_model(X_train, y_train)

# Save model as dictionary
model_package = {
    'model': model,
    'encoder': encoder,
    'label_binarizer': lb, 
    'scaler': scaler
}
joblib.dump(model_package, 'model/model_package.pkl')


# Process the test data with the process_data function.
X_test, y_test, _, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb, scaler=scaler
)

# Predictions and evaluation
preds = inference(model, X_test)
precision, recall, fbeta, f1 = compute_model_metrics(y_test, preds)
print(f"f1: {f1}, precision: {precision}, recall: {recall}, fbeta: {fbeta}")

# Save the score to a file for traceability
with open('output/score.json', 'w') as score_file:
    json.dump(
        {"F1 score": f1, 
         "precision": precision, 
         "recall": recall, 
         "fbeta": fbeta
         }, 
        score_file)