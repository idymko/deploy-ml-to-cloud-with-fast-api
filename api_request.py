import requests
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ML Census Income Prediction")
    parser.add_argument('--url', 
                        type=str, 
                        help='URL to server', 
                        default="http://127.0.0.1:8000/ml/",
                        required=False)
    
    args = parser.parse_args()
    
    payload = {
        "age": 52, 
        "workclass": "Self-emp-not-inc", 
        "fnlgt": 289436, 
        "education": "Doctorate", 
        "education-num": 16, 
        "marital-status": 
            "Married-civ-spouse", 
            "occupation": "Prof-specialty", 
            "relationship": "Husband", 
            "race": "White", 
            "sex": "Male", 
            "capital-gain": 0, 
            "capital-loss": 0, 
            "hours-per-week": 60, 
            "native-country": "United-States"}
    
    response = requests.post(args.url, json=payload)
    
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())