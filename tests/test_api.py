
import pytest
from fastapi.testclient import TestClient

from api.api import app

# Create a test client for the FastAPI application
client = TestClient(app)

# Example machine operating conditions
VALID_INPUT = {
    "air_temp": 298.1,
    "process_temp": 308.6,
    "rpm": 1500,
    "torque": 40,
    "tool_wear": 100
}


# Test 1: Valid prediction request
def test_valid_prediction():
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["failure_probability"] <= 1
    assert result["failure"] in [0, 1]
    assert 0 <= result["threshold"] <= 1


# Test 2: Missing required feature
def test_missing_feature():
    invalid_input = VALID_INPUT.copy()
    invalid_input.pop("rpm")

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


# Test 3: Invalid negative feature values
@pytest.mark.parametrize(
    "field,value",
    [
        ("air_temp", -10),
        ("process_temp", -10),
        ("rpm", -100),
        ("torque", -5),
        ("tool_wear", -1),
    ],
)
def test_negative_values(field, value):
    invalid_input = VALID_INPUT.copy()
    invalid_input[field] = value

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


# Test 4: Prediction consistency
def test_prediction_consistency():
    response1 = client.post("/predict", json=VALID_INPUT)
    response2 = client.post("/predict", json=VALID_INPUT)

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()
