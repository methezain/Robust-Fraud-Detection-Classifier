"""
FastAPI deployment for fraud detection model.
Production-ready REST API for real-time fraud detection.
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from model import FraudDetectionModel
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions",
    version="1.0.0"
)

# Global model instance
fraud_service = None

# Pydantic models for request/response
class Transaction(BaseModel):
    step: int
    type: str  
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

class FraudPrediction(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: str

class BatchTransaction(BaseModel):
    transactions: List[Transaction]

class BatchPrediction(BaseModel):
    predictions: List[FraudPrediction]
    summary: Dict[str, int]

class FraudDetectionService:
    """
    Production service for fraud detection.
    Handles model loading and real-time predictions.
    """
    
    def __init__(self, model_path='fraud_detection_model.joblib'):
        """Initialize the service with a trained model."""
        self.detector = FraudDetectionModel()

        try:
            self.detector.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_single_transaction(self, transaction_dict):
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction_dict: Dictionary containing transaction features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            df = pd.DataFrame([transaction_dict])
            
            df_processed = self.detector.preprocess_data(df)
            
            prediction, probability = self.detector.predict_fraud(df_processed)
            
            return { 
                'is_fraud': bool(prediction[0]),
                'fraud_probability': float(probability[0]),
                'risk_level': self._get_risk_level(probability[0]),
                'confidence': self._get_confidence_level(probability[0])
            }
            
        except Exception as e:
            print(f"Error predicting transaction: {e}")
            raise
    
    def predict_batch_transactions(self, transactions_list):
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions_list: List of transaction dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(transactions_list)
            
            # Preprocess transactions
            df_processed = self.detector.preprocess_data(df)
            
            # Make predictions
            predictions, probabilities = self.detector.predict_fraud(df_processed)
            
            # Format results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'is_fraud': bool(pred),
                    'fraud_probability': float(prob),
                    'risk_level': self._get_risk_level(prob),
                    'confidence': self._get_confidence_level(prob)
                })
            
            return results
            
        except Exception as e:
            print(f"Error predicting batch transactions: {e}")
            raise
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level."""
        if probability >= 0.8:
            return 'HIGH'
        elif probability >= 0.5:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _get_confidence_level(self, probability):
        """Get confidence level for the prediction."""
        confidence_score = max(probability, 1 - probability) 

        if confidence_score >= 0.9:
            return 'VERY_HIGH'
        elif confidence_score >= 0.7:
            return 'HIGH'
        elif confidence_score >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the fraud detection service on startup."""
    global fraud_service
    try:
        fraud_service = FraudDetectionService('fraud_detection_model.joblib')
        print("Fraud Detection API is ready!")
    except Exception as e:
        print(f"Failed to initialize fraud detection service: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": fraud_service is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Fraud prediction result
    """
    if fraud_service is None:
        raise HTTPException(status_code=503, detail="Fraud detection service not available")
    
    try:
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = fraud_service.predict_single_transaction(transaction_dict)
        
        return FraudPrediction(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPrediction)
async def predict_fraud_batch(batch: BatchTransaction):
    """
    Predict fraud for multiple transactions.
    
    Args:
        batch: Batch of transactions
        
    Returns:
        Batch prediction results with summary
    """
    if fraud_service is None:
        raise HTTPException(status_code=503, detail="Fraud detection service not available")
    
    try:
        # Convert Pydantic models to list of dicts
        transactions_list = [t.dict() for t in batch.transactions]
        
        # Make predictions
        results = fraud_service.predict_batch_transactions(transactions_list)
        
        # Create summary
        summary = {
            "total_transactions": len(results),
            "predicted_fraud": sum(1 for r in results if r['is_fraud']),
            "predicted_legitimate": sum(1 for r in results if not r['is_fraud']),
            "high_risk": sum(1 for r in results if r['risk_level'] == 'HIGH'),
            "medium_risk": sum(1 for r in results if r['risk_level'] == 'MEDIUM'),
            "low_risk": sum(1 for r in results if r['risk_level'] == 'LOW')
        }
        
        predictions = [FraudPrediction(**result) for result in results]
        
        return BatchPrediction(predictions=predictions, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if fraud_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "RandomForestClassifier with SMOTE",
        "features_expected": "Transaction amount, balances, type, and engineered features",
        "preprocessing": "StandardScaler for numerical, one-hot encoding for categorical",
        "sampling_strategy": "SMOTE with 0.1 ratio",
        "status": "loaded and ready"
    }


def create_sample_transaction():
    """Create a sample transaction for testing."""
    return {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 9000.60,
        'oldbalanceOrg': 9000.60,
        'newbalanceOrig': 0.00,
        'oldbalanceDest': 0.00,
        'newbalanceDest': 0.00
    }


if __name__ == "__main__":
    """Run the FastAPI server."""
    print("Starting Fraud Detection API...")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Alternative docs: http://127.0.0.1:8000/redoc")
    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=4000,
        reload=True,
        log_level="info"
    )