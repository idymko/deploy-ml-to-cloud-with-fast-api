from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data import process_data
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def get_inference_pipeline(rf_config, y):
	"""
	Create a pipeline including:
	* data preprocessing for categorical and numerical values
 	* model for training
	"""
  	# Encode the labels
	label_encoder = LabelBinarizer()
	y_encoded = label_encoder.fit_transform(y)
 
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
	sk_pipe = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("random_forest", random_forest)
		]
	)
 
	return sk_pipe, y_encoded
 
def train_model(X_train, y_train):
	"""
	Trains a machine learning model and returns it.

	Inputs
	------
	X_train : np.ndarray
		Training data.
	y_train : np.ndarray
		Labels.
	Returns
	-------
	model : RandomForestClassifier
		Trained machine learning model.
	"""
	pipeline, y_encoded_train = get_inference_pipeline(X_train, y_train)
	pipeline.fit(X_train, y_encoded_train)
	
	return pipeline


def compute_model_metrics(y, preds):
	"""
	Validates the trained machine learning model using precision, recall, and F1.

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
	f1: float
	"""
	fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
	precision = precision_score(y, preds, zero_division=1)
	recall = recall_score(y, preds, zero_division=1)
	f1 = f1_score(y, preds)
	return precision, recall, fbeta, f1


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
	loaded_package = joblib.load(path) # 'model_package.pkl'

	# Access the components
	model = loaded_package['model']
	encoder = loaded_package['encoder']
	lb = loaded_package['label_binarizer']
	scaler = loaded_package['scaler']
	
	return model, encoder, lb, scaler

def performance_per_slice(data, model):
	"""
	Write a function that outputs the performance of the model on slices of the data.

	Suggestion: for simplicity, the function can 
		just output the performance on slices of just the categorical features.
	"""
	
	model, encoder, lb, scaler = load_model_package("model/model_package.pkl")
 
	X_eval, y_eval, _, _, _ = process_data(
		data, categorical_features=[], label="salary", training=False, 
		encoder=encoder, lb=lb, scaler=scaler
	)
 	
	cat_features = ["education"]
	for feature in cat_features:
		print(f"\nfeature: {feature}")
		for cls in data[feature].unique():
			data_slice = data[data[feature]==cls]
			preds = inference(model, X_eval)
			precision, recall, fbeta, f1 = compute_model_metrics(y_eval, preds)
			print(f"f1: {f1}, precision: {precision}, recall: {recall}, fbeta: {fbeta}")
	

# if __name__ == "__main__":
	
#     y_true =    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
#     preds =     [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
#     fbeta, precision, recall, f1 = compute_model_metrics(y_true, preds)
	
#     print(f"f1: {f1}, precision: {precision}, recall: {recall}, fbeta: {fbeta}")