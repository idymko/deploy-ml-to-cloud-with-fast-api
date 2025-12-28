# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Dmytro Kysylychyn created the model. It is a Random Forest Classifier from scikit-learn using the hyperparameters:
 * n_estimators: 100
 * max_depth: 10
 * random_state: 42.

## Intended Use
This model should be used to predit the salary level (>50k or <=50k) based on several parameters such as: 
 * age
 * workclass
 * fnlgt
 * education
 * education-num
 * marital-status
 * occupation
 * relationship
 * race
 * sex
 * capital-gain
 * capital-loss
 * hours-per-week
 * native-country


## Training Data
A classification model was trained on publicly available Census Bureau data: https://archive.ics.uci.edu/dataset/20/census+income.

## Evaluation Data
Evaluation data can found in `data\test_data.csv`.

## Metrics
* The model was evaluated by using precision,recall,fbeta.
* Evaluation on `data\test_data.csv` is as follows: 
    * precision:0.8026
    * recall:0.5461
    * fbeta:0.6500.

## Ethical Considerations
Model shows lower precision for specific races such as Asian-Pac-Islander, which could be due to inbalanced dataset.

## Caveats and Recommendations
The limitation of the model is the age of training dataset (1994 Census) meaning a potential distribution shift in modern income predictions.