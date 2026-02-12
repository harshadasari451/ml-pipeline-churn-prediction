"""
Tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
        assert 'docs_url' in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'model_loaded' in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing."""
        return {
            "customer": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.50,
                "TotalCharges": 846.00
            }
        }
    
    def test_predict_endpoint_without_model(self, client, sample_customer):
        """Test prediction endpoint when model is not loaded."""
        response = client.post("/predict", json=sample_customer)
        
        # Should return 503 if model not loaded
        assert response.status_code in [503, 500]
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_data = {
            "customer": {
                "gender": "Invalid",
                "SeniorCitizen": 5,  # Should be 0 or 1
            }
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {
            "customer": {
                "gender": "Female",
                "SeniorCitizen": 0,
            }
        }
        
        response = client.post("/predict", json=incomplete_data)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_batch_predict_endpoint_without_model(self, client, sample_customer):
        """Test batch prediction endpoint when model is not loaded."""
        batch_data = {
            "customers": [sample_customer["customer"], sample_customer["customer"]]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        # Should return 503 if model not loaded
        assert response.status_code in [503, 500]
    
    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty customer list."""
        batch_data = {
            "customers": []
        }
        
        response = client.post("/predict/batch", json=batch_data)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/predict/model/info")
        
        # Should return 404 if model not loaded, or 200 if loaded
        assert response.status_code in [404, 200, 500]


class TestAPIValidation:
    """Tests for API input validation."""
    
    def test_gender_validation(self, client):
        """Test gender field validation."""
        data = {
            "customer": {
                "gender": "Other",  # Invalid
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0
            }
        }
        
        response = client.post("/predict", json=data)
        assert response.status_code == 422
    
    def test_senior_citizen_validation(self, client):
        """Test SeniorCitizen field validation."""
        data = {
            "customer": {
                "gender": "Male",
                "SeniorCitizen": 2,  # Invalid - should be 0 or 1
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0
            }
        }
        
        response = client.post("/predict", json=data)
        assert response.status_code == 422
    
    def test_negative_charges_validation(self, client):
        """Test validation of negative charges."""
        data = {
            "customer": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": -50.0,  # Invalid
                "TotalCharges": 600.0
            }
        }
        
        response = client.post("/predict", json=data)
        assert response.status_code == 422


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert 'openapi' in data
        assert 'paths' in data
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
