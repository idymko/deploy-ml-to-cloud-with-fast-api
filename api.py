from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Annotated, List
from src.model import load_model_package, inference
import yaml
import pandas as pd

valid_workclasses = ["State-gov", "Self-emp-not-inc", "Private", "Federal-gov", "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked"]
valid_education = [
            "Bachelors", "HS-grad", "11th", "Masters", "9th", 
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", 
            "Doctorate", "Prof-school", "5th-6th", "10th", 
            "1st-4th", "Preschool", "12th"
        ]
valid_marital_status = [
            "Never-married", "Married-civ-spouse", "Divorced", 
            "Married-spouse-absent", "Separated", "Married-AF-spouse", 
            "Widowed"
        ]
valid_occupation = [
            "Adm-clerical", "Exec-managerial", "Handlers-cleaners", 
            "Prof-specialty", "Other-service", "Sales", "Craft-repair", 
            "Transport-moving", "Farming-fishing", "Machine-op-inspct", 
            "Tech-support", "?", "Protective-serv", "Armed-Forces", 
            "Priv-house-serv"
        ]
valid_relationship = [
            "Not-in-family", "Husband", "Wife", "Own-child", 
            "Unmarried", "Other-relative"
        ]
valid_race = [
            "White", "Black", "Asian-Pac-Islander", 
            "Amer-Indian-Eskimo", "Other"
        ]
valid_sex = ["Male", "Female"]
valid_native_countries = [
            "United-States", "Cuba", "Jamaica", "India", "?", 
            "Mexico", "South", "Puerto-Rico", "Honduras", "England", 
            "Canada", "Germany", "Iran", "Philippines", "Italy", 
            "Poland", "Columbia", "Cambodia", "Thailand", "Ecuador", 
            "Laos", "Taiwan", "Haiti", "Portugal", "Dominican-Republic", 
            "El-Salvador", "France", "Guatemala", "China", "Japan", 
            "Yugoslavia", "Peru", "Outlying-US(Guam-USVI-etc)", 
            "Scotland", "Trinidad&Tobago", "Greece", "Nicaragua", 
            "Vietnam", "Hong", "Ireland", "Hungary", "Holand-Netherlands"
        ]

class Data(BaseModel):
    
    # Using Pydantic Validator
    
    age: Annotated[int, Field(gt=0, lt=122, description="Must be a positive number")]
    workclass: Annotated[str, Field(description="Workclass", examples = valid_workclasses)]
    fnlgt: Annotated[int, Field(gt=0, description="Must be a positive number")]
    education: Annotated[str, Field(description="Education", examples = valid_education)]
    education_num: Annotated[int, Field(ge=0, description="Must not be a negative number", alias = "education-num")]
    marital_status: Annotated[str, Field(description="Marital status", alias='marital-status', examples = valid_marital_status)]
    occupation: Annotated[str, Field(description="Occupation type", examples = valid_occupation)]
    relationship: Annotated[str, Field(description="Relationship status", examples = valid_relationship)]
    race: Annotated[str, Field(description="Race category", examples = valid_race)]
    sex: Annotated[str, Field(description="Gender", examples = valid_sex)]
    capital_gain: Annotated[int, Field(ge=0, description="Must not be a negative number", alias = "capital-gain")]
    capital_loss: Annotated[int, Field(ge=0, description="Must not be a negative number", alias = "capital-loss")]
    hours_per_week: Annotated[int, Field(ge=0, description="Must not be a negative number", alias = "hours-per-week")]
    native_country: Annotated[str, Field(description="Country of origin", alias='native-country', examples = valid_native_countries)]
    
    @field_validator('workclass')
    def check_workclass(v):
        if v not in valid_workclasses:
            raise ValueError(f"Invalid workclass '{v}'. Must be one of {valid_workclasses}.")
        return v
    
    @field_validator('education')
    def check_education(v):
        if v not in valid_education:
            raise ValueError(f"Invalid education '{v}'. Must be one of {valid_education}.")
        return v

    @field_validator('marital_status')
    def check_marital_status(v):
        if v not in valid_marital_status:
            raise ValueError(f"Invalid marital status '{v}'. Must be one of {valid_marital_status}.")
        return v

    @field_validator('occupation')
    def check_occupation(v):
        if v not in valid_occupation:
            raise ValueError(f"Invalid occupation '{v}'. Must be one of {valid_occupation}.")
        return v

    @field_validator('relationship')
    def check_relationship(v):
        if v not in valid_relationship:
            raise ValueError(f"Invalid relationship '{v}'. Must be one of {valid_relationship}.")
        return v

    @field_validator('race')
    def check_race(v):
        if v not in valid_race:
            raise ValueError(f"Invalid race '{v}'. Must be one of {valid_race}.")
        return v

    @field_validator('sex')
    def check_sex(v):
        if v not in valid_sex:
            raise ValueError(f"Invalid sex '{v}'. Must be one of {valid_sex}.")
        return v

    @field_validator('native_country')
    def check_native_country(v):
        if v not in valid_native_countries:
            raise ValueError(f"Invalid native country '{v}'. Must be one of {valid_native_countries}.")
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "age": 29,
                "workclass": "Private",
                "fnlgt": 133937,
                "education": "Doctorate",
                "education-num": 16,
                "marital-status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Own-child",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
            ]
    })

app = FastAPI(
    title = "ML Census Income Prediction",
    description = "An API to feed data into ML model",
    version = "1.0.0",
    openapi_version = "3.1.0"
)

# # GET must be on the root domain and give a greeting
@app.get("/")
async def get_greeting():
    return "Greetings!"

# # POST on a different path that does model inference.
@app.post("/ml/")
async def ml_inference(payload: Data):
    # Validation happens automatically through Pydantic
    # Load parameters
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # Load in the data.    
    model_path = params['evaluate']['model_path']

    # Evaluate overall model
    model, label_encoder = load_model_package(model_path)
    
    df = pd.DataFrame([payload.model_dump()])
    df.columns = df.columns.str.replace("_", "-", regex=False)
    preds = inference(model, df)
    return {"prediction": label_encoder.inverse_transform(preds).tolist()[0], "input": payload}

# # Usage: `uvicorn api:app --reload`