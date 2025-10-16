# Fraud Detection Model

A production-ready fraud detection system using machine learning to identify suspicious financial transactions.

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Train the Model

```powershell
python model.py
```

### 3. Test the Model

```powershell
python test_model.py
```

### 4. Use for Predictions

```powershell
python deploy.py
```

## 📋 What's Fixed

### Major Issues Resolved:
- ✅ **Missing imports** - Added all required libraries
- ✅ **Undefined variables** - Proper variable initialization and flow
- ✅ **Inconsistent preprocessing** - Streamlined single preprocessing approach
- ✅ **Poor function organization** - Class-based structure for better maintainability
- ✅ **Mixed responsibilities** - Clear separation of concerns

### Improvements Made:
- 🔧 **Simplified architecture** - Easy to understand and maintain
- 🔧 **Better error handling** - Comprehensive try-catch blocks
- 🔧 **Production ready** - Deployment service included
- 🔧 **Clear function names** - Self-explanatory method names
- 🔧 **Better feature engineering** - More meaningful features
- 🔧 **Robust data handling** - Handles edge cases properly

## 📁 File Structure

```
fraud_detection/
├── model.py          # Main model class with training pipeline
├── test_model.py     # Testing script for development
├── deploy.py         # Production deployment service
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## 🛠 Model Features

### Data Processing:
- Automatic data loading from Kaggle
- Feature engineering (balance changes, error flags, ratios)
- One-hot encoding for transaction types
- Proper handling of missing values

### Model Pipeline:
- StandardScaler for numerical features
- SMOTE for handling imbalanced data
- RandomForest classifier with balanced classes
- Complete scikit-learn pipeline

### Evaluation Metrics:
- Classification report
- AUC-ROC score
- Confusion matrix
- Risk level categorization

## 🎯 Usage Examples

### Training a New Model:
```python
from model import FraudDetectionModel

# Initialize and train
detector = FraudDetectionModel(random_state=42)
df = detector.load_data()
df_processed = detector.preprocess_data(df)
X_train, X_test, y_train, y_test = detector.split_data(df_processed)
detector.train_model(X_train, y_train)
detector.save_model('my_model.joblib')
```

### Making Predictions:
```python
from deploy import FraudDetectionService

# Load trained model
service = FraudDetectionService('fraud_detection_model.joblib')

# Predict single transaction
transaction = {
    'step': 1,
    'type': 'TRANSFER',
    'amount': 5000.0,
    'oldbalanceOrg': 5000.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0
}

result = service.predict_single_transaction(transaction)
print(f"Fraud probability: {result['fraud_probability']:.2%}")
```

## 📊 Model Performance

The model uses:
- **RandomForestClassifier** with 100 estimators
- **SMOTE** oversampling for class balance
- **StandardScaler** for feature normalization
- **Cross-validation** ready architecture

Expected performance:
- High precision for fraud detection
- Balanced recall to catch most fraud cases
- Fast inference time for real-time predictions

## 🔧 Customization

### Adjusting Model Parameters:
```python
# In model.py, modify the RandomForestClassifier
RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=5,   # More conservative splits
    random_state=42
)
```

### Adding New Features:
```python
# In engineer_features method, add:
df['new_feature'] = df['amount'] / df['step']
```

## 🚨 Important Notes

1. **Data Privacy**: Ensure compliance with data protection regulations
2. **Model Monitoring**: Regularly retrain with new data
3. **Threshold Tuning**: Adjust probability thresholds based on business needs
4. **Feature Drift**: Monitor for changes in data distribution

## 📞 Support

For issues or questions:
1. Check the error messages in console output
2. Verify all dependencies are installed correctly
3. Ensure the Kaggle dataset is accessible
4. Review the logs for detailed error information

## 📈 Next Steps

1. **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
2. **Feature Selection**: Implement feature importance analysis
3. **Model Comparison**: Try different algorithms (XGBoost, LightGBM)
4. **Real-time API**: Deploy as REST API using Flask/FastAPI
5. **Monitoring Dashboard**: Create visualization for model performance