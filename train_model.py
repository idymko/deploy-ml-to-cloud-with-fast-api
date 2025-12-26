# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd 
import joblib
import json 

import yaml

# Add the necessary imports for the starter code.
from src.model import train_model, compute_model_metrics, inference

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

data_path = params['train']['data_path']
n_estimators = params['train']['n_estimators']
max_depth = params['train']['max_depth']
random_state  = params['train']['random_state']

# Load in the data.  
X = pd.read_csv(data_path)
y = X.pop("salary")         # separate label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Dump some data for testing (ideally we should have a separate set). 
test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
test_data.to_csv('data/test_data.csv', index=False)

# Train model
model, label_encoder = train_model(X_train=X_train, 
                                   y_train=y_train,
                                   rf_config = {
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth,
                                        'random_state': random_state
                                    })

# Save model as dictionary
artifact = {
    "model": model,
    "label_encoder": label_encoder
}
joblib.dump(artifact, 'model/model.pkl')

# Predictions and evaluation
preds = inference(model, X_test)
y_test = label_encoder.transform(y_test.values).ravel()
score = model.score(X_test,y_test)
print(f"score: {score}")

precision, recall, fbeta = compute_model_metrics(y_test, preds)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")

# Save the score to a file for traceability
with open('output/score.json', 'w') as score_file:
    json.dump(
        {"precision": precision, 
         "recall": recall, 
         "fbeta": fbeta
         }, 
        score_file)