from fastapi.testclient import TestClient
import pytest

from api import app # import our app
client = TestClient(app)

@pytest.mark.anyio
async def test_get_greeting():
    
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Greetings!"

@pytest.mark.anyio
async def test_ml_inference_success_0():
    input_data = {"age": 29, 
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

    
    r = client.post("/ml", json=input_data)
    
    assert r.status_code == 200
    response_data = r.json()
    assert input_data == response_data["input"]
    assert response_data["prediction"] == "<=50K"

@pytest.mark.anyio
async def test_ml_inference_success_1():
    input_data = {"age": 52,
                            "workclass": "Self-emp-not-inc",
                            "fnlgt": 289436,
                            "education": "Doctorate",
                            "education-num": 16,
                            "marital-status": "Married-civ-spouse",
                            "occupation": "Prof-specialty",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 0,
                            "capital-loss": 0,
                            "hours-per-week": 60,
                            "native-country": "United-States"
                        }
    
    r = client.post("/ml", json=input_data)
    
    assert r.status_code == 200
    response_data = r.json()
    assert input_data == response_data["input"]
    assert response_data["prediction"] == ">50K"
    
@pytest.mark.anyio
async def test_ml_inference_failure():
    input_data = {"age": 52,
                            "workclass": "Self-emp-not-inc",
                            "fnlgt": 289436,
                            "education": "Doctorate",
                            "education-num": 16,
                            "marital-status": "Married-civ-spouse",
                            "occupation": "Prof-specialty",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 0,
                            "capital-loss": 0,
                            "hours-per-week": 60,
                            "native-country": "United-X"
                        }
    
    r = client.post("/ml", json=input_data)
    
    assert r.status_code == 422