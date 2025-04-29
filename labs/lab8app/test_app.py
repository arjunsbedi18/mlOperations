import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

sample_data = {"features": [13.24, 2.59, 2.87, 21.0, 118.0, 2.8, 2.69, 
        0.39, 1.82, 4.32, 1.04, 2.93, 735.0]}

# Make POST request to the API
try:
    response = requests.post(url, json=sample_data)
    
    # Print status code and response
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON:")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredicted Wine Class: {result['prediction']}")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(f"Error message: {response.text}")
        
except requests.RequestException as e:
    print(f"Request error: {e}")
except Exception as e:
    print(f"Error: {e}") 