from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd 

def get_pipeline(rf_config):
	"""
	Create a pipeline including:
	* data preprocessing for categorical and numerical values
 	* model for training
  
  	Inputs
	------
	rf_config: dict
		config for RandomForestClassifier
	
	Outputs
	------
	model_pipeline: Pipeline
		preprocessing and model pipeline
  
	"""
 
	# Categorical features
	categorical_features = [
				"workclass",
				"education",
				"marital-status",
				"occupation",
				"relationship",
				"race",
				"sex",
				"native-country",
			]
	categorical_features_preproc = make_pipeline(
			OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
		)
 
 	# Numerical features
	numerical_features = ['age',
				'fnlgt',
				'education-num',
				'capital-gain',
				'capital-loss',
				'hours-per-week'
			]
	numerical_features_preproc = make_pipeline(
		StandardScaler()
	)
 
 	# Let's put everything together
	preprocessor = ColumnTransformer(
		transformers=[
			("categorical", categorical_features_preproc, categorical_features),
			("numerical", numerical_features_preproc, numerical_features),
		],
		remainder="drop",  # This drops the columns that we do not transform
	)
 
	# Create random forest
	random_forest = RandomForestClassifier(**rf_config)

	# Create the inference pipeline.
	model_pipeline = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("random_forest", random_forest)
		]
	)
 
	return model_pipeline
 
def train_model(X_train, y_train, rf_config):
	"""
	Trains a machine learning model and returns it.

	Inputs
	------
	X_train : np.ndarray
		Training data.
	y_train : np.ndarray
		Labels.
	rf_config: dict
		config for RandomForestClassifier
	Returns
	-------
	model : RandomForestClassifier
		Trained machine learning model.
  	label_encoder : LabelBinarizer
		Label encoder
	"""
	
	# Encode the labels
	label_encoder = LabelBinarizer()
	y_train_encoded = label_encoder.fit_transform(y_train.values).ravel()
 
	model = get_pipeline(rf_config)
	model.fit(X_train, y_train_encoded)
	
	return model, label_encoder


def compute_model_metrics(y, preds):
	"""
	Validates the trained machine learning model using precision, recall.

	Inputs
	------
	y : np.ndarray
		Known labels, binarized.
	preds : np.ndarray
		Predicted labels, binarized.
	Returns
	-------
	precision : float
	recall : float
	fbeta : float
	"""
	fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
	precision = precision_score(y, preds, zero_division=1)
	recall = recall_score(y, preds, zero_division=1)
	return precision, recall, fbeta

def inference(model, X):
	""" Run model inferences and return the predictions.

	Inputs
	------
	model : RandomForestClassifier
		Trained machine learning model.
	X : np.ndarray
		Data used for prediction.
	Returns
	-------
	preds : np.ndarray
		Predictions from the model.
	"""
	preds = model.predict(X)
	return preds

def load_model_package(path):
	# Load the model package
	loaded_package = joblib.load(path)

	# Access the components
	model = loaded_package['model']
	label_encoder = loaded_package['label_encoder']
	
	return model, label_encoder

def performance_per_slice(data, model, label_encoder, feature):
	"""
	Write a function that outputs the performance of the model on slices of the data.

	Suggestion: for simplicity, the function can 
		just output the performance on slices of just the categorical features.
	Inputs
	------
	data : pd.DataFrame
	model : Pipeline
	label_encoder : label encoder for y values
	feature : str, name of the feature on which to perform the model evaluation
	"""

	y_test = data.pop("salary")
	y_test = label_encoder.transform(y_test.values).ravel()

	results = []  # List to collect results

	for cls_ in data[feature].unique():
		cls_indies = data[feature]==cls_
		data_slice = data[cls_indies]
		preds = inference(model, data_slice)
		precision, recall, fbeta = compute_model_metrics(y_test[cls_indies], preds)

		# Append results to the list
		results.append({
			'feature': feature,
			'class': cls_,
			'precision': precision,
			'recall': recall,
			'fbeta': fbeta
		})
	
	results_df = pd.DataFrame(results)
	results_df = results_df.sort_values(by='precision', ascending=True)
	results_df = results_df.reset_index(drop=True)
	results_df.to_csv('output/slice_output.txt')
	print("Model metrics for slices")
	print(results_df.head(10))