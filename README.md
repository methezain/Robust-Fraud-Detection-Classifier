# 🛡️ Real-Time Fraud Detection System

Machine learning-powered API for detecting fraudulent financial transactions with instant predictions.

---

## 🎯 Problem Statement

Financial fraud costs billions annually, with fraudulent transactions often going undetected until significant damage occurs. Traditional rule-based systems struggle with:
- **High false-positive rates** leading to legitimate transaction blocks
- **Inability to adapt** to evolving fraud patterns
- **Manual review delays** causing poor customer experience
- **Imbalanced datasets** where fraud cases are rare (~0.1%)

---

## � Solution

A **machine learning microservice** that analyzes transactions in real-time and provides instant fraud predictions with:
- **Probability scores** (0-100%) indicating fraud likelihood
- **Risk classifications** (HIGH/MEDIUM/LOW/MINIMAL)
- **REST API endpoints** for single and batch predictions
- **Web interface** for easy testing and visualization

---

## ⚙️ Technical Overview

### Architecture
```
PaySim Dataset → Feature Engineering → SMOTE Balancing → Random Forest → FastAPI → Prediction
```

### Key Components
- **Model**: Random Forest classifier (100 estimators) with SMOTE oversampling
- **Features**: Balance changes, transaction ratios, error flags, one-hot encoded types
- **API**: FastAPI with CORS support, health monitoring, and batch processing
- **Frontend**: Responsive web UI with real-time health checks and drag-drop uploads
- **Deployment**: Docker containerized for scalability

### Performance
- **Response Time**: ~50-200ms per prediction
- **Features**: 17 engineered features from 7 raw inputs
- **Handles**: TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT transactions

---

## 📈 Impact

✅ **Instant Decision Making** - Block suspicious transactions before completion  
✅ **Reduced False Positives** - ML learns patterns vs rigid rules  
✅ **Scalable** - Handle thousands of requests via containerized deployment  
✅ **Transparency** - Probability scores and confidence levels for review  
✅ **Cost Effective** - Automated detection reduces manual review workload  

---

## 🚀 Quick Start

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python app.py

# Access at http://localhost:4000
```

### Run with Docker
```bash
# Build and run
docker build -t fraud-detection-api .
docker run -d -p 4000:4000 fraud-detection-api

# Or use docker-compose
docker-compose up -d
```

### Test API
```bash
# Single prediction
python test_single_transaction.py

# Batch predictions
python test_batch_transactions.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | System status |
| `/predict` | POST | Single transaction |
| `/predict/batch` | POST | Batch transactions |
| `/model/info` | GET | Model details |

---

## �️ Tech Stack

**ML/Data**: scikit-learn • pandas • numpy • imbalanced-learn  
**API**: FastAPI • uvicorn • pydantic  
**Frontend**: HTML5 • CSS3 • JavaScript (Vanilla)  
**Deployment**: Docker • Docker Compose  

---

## � Project Structure

```
├── app.py                          # FastAPI application
├── model.py                        # ML model training
├── fraud_detection_model.joblib    # Trained model
├── static/                         # Web frontend
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── test_single_transaction.py      # API test script
├── test_batch_transactions.py      # Batch test script
├── Dockerfile                      # Container config
├── docker-compose.yml              # Orchestration
└── requirements.txt                # Dependencies
```

---

## 📄 License

MIT License - Feel free to use and modify.