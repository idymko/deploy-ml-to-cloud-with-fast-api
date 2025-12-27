from fastapi.testclient import TestClient
import warnings 
import pytest

from api import app # import our app
client = TestClient(app)

@pytest.mark.anyio
async def test_get_greeting():
    
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Greetings!"
    
@pytest.mark.anyio
async def test_ml_inference_sucess():
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