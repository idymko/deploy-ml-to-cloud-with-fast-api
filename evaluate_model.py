from src.model import load_model_package, inference, compute_model_metrics, performance_per_slice
import pandas as pd 
import yaml

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load in the data.    
data_path = params['evaluate']['data_path']
model_path = params['evaluate']['model_path']


# Evaluate overall model
model, label_encoder = load_model_package(model_path)
eval_data = pd.read_csv(data_path)
X_test = eval_data.copy()
y_test = X_test.pop("salary")
y_test = label_encoder.transform(y_test.values).ravel()
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"model metrics for data_path: {data_path}\nprecision:{precision:.4f},recall:{recall:.4f},fbeta:{fbeta:.4f}\n")

# Evaluate model for difference slices
performance_per_slice(eval_data.copy(), model, label_encoder, 'race')
performance_per_slice(eval_data.copy(), model, label_encoder, 'education')