from src.model import load_model_package, inference, compute_model_metrics
from src.data import process_data
import pandas as pd 
import yaml

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load in the data.    
data_path = params['evaluate']['data_path']
model_path = params['evaluate']['model_path']



model, encoder, lb, scaler = load_model_package(model_path)
eval_data = pd.read_csv(data_path)

X_eval, y_eval, _, _, _ = process_data(
		eval_data, categorical_features=[], label="salary", training=False, 
		encoder=encoder, lb=lb, scaler=scaler
	)
preds = inference(model, X_eval)

precision, recall, fbeta, f1 = compute_model_metrics(y_eval, preds)
print(f"f1: {f1}, precision: {precision}, recall: {recall}, fbeta: {fbeta}")