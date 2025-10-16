// API Base URL - Update this if your API runs on a different port
const API_BASE_URL = 'http://127.0.0.1:4000';

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    setupEventListeners();
    loadModelInfo();
    
    // Check health every 30 seconds
    setInterval(checkAPIHealth, 30000);
});

// Setup event listeners
function setupEventListeners() {
    // Single transaction form
    document.getElementById('transactionForm').addEventListener('submit', handleSingleTransaction);
    
    // File upload
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/json') {
            handleFile(file);
        } else {
            alert('Please upload a JSON file');
        }
    });
}

// Check API health
async function checkAPIHealth() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        const timestamp = new Date(data.timestamp).toLocaleTimeString();
        
        if (data.status === 'healthy' && data.model_loaded) {
            statusDot.className = 'status-dot healthy';
            statusText.textContent = `✓ System Online - Model Ready (Last checked: ${timestamp})`;
        } else {
            statusDot.className = 'status-dot error';
            statusText.textContent = `⚠ System Online - Model Not Loaded (${timestamp})`;
        }
    } catch (error) {
        statusDot.className = 'status-dot error';
        statusText.textContent = '✗ API Unavailable - Please start the server';
        console.error('Health check failed:', error);
    }
}

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}Tab`).classList.add('active');
    event.target.classList.add('active');
    
    // Load model info when switching to that tab
    if (tabName === 'model') {
        loadModelInfo();
    }
}

// Handle single transaction prediction
async function handleSingleTransaction(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const transaction = {
        step: parseInt(formData.get('step')),
        type: formData.get('type'),
        amount: parseFloat(formData.get('amount')),
        oldbalanceOrg: parseFloat(formData.get('oldbalanceOrg')),
        newbalanceOrig: parseFloat(formData.get('newbalanceOrig')),
        oldbalanceDest: parseFloat(formData.get('oldbalanceDest')),
        newbalanceDest: parseFloat(formData.get('newbalanceDest'))
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transaction)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        displaySingleResult(result);
    } catch (error) {
        alert('Error analyzing transaction: ' + error.message);
        console.error('Prediction error:', error);
    }
}

// Display single transaction result
function displaySingleResult(result) {
    const resultsCard = document.getElementById('resultsCard');
    const resultsContent = document.getElementById('resultsContent');
    
    const isFraud = result.is_fraud;
    const probability = (result.fraud_probability * 100).toFixed(2);
    
    resultsContent.innerHTML = `
        <div class="fraud-alert ${isFraud ? 'danger' : 'success'}">
            ${isFraud ? '⚠️ FRAUD DETECTED' : '✓ TRANSACTION LEGITIMATE'}
        </div>
        
        <div class="result-grid">
            <div class="result-item">
                <h3>Fraud Probability</h3>
                <div class="value">${probability}%</div>
            </div>
            
            <div class="result-item">
                <h3>Risk Level</h3>
                <div class="value">
                    <span class="risk-badge risk-${result.risk_level}">${result.risk_level}</span>
                </div>
            </div>
            
            <div class="result-item">
                <h3>Confidence</h3>
                <div class="value">${result.confidence.replace('_', ' ')}</div>
            </div>
            
            <div class="result-item">
                <h3>Decision</h3>
                <div class="value">${isFraud ? 'BLOCK' : 'APPROVE'}</div>
            </div>
        </div>
    `;
    
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Handle file upload
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Process uploaded file
function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const content = e.target.result;
            document.getElementById('jsonInput').value = content;
        } catch (error) {
            alert('Error reading file: ' + error.message);
        }
    };
    reader.readAsText(file);
}

// Analyze batch transactions
async function analyzeBatch() {
    const jsonInput = document.getElementById('jsonInput').value.trim();
    
    if (!jsonInput) {
        alert('Please provide transaction data');
        return;
    }
    
    try {
        const transactions = JSON.parse(jsonInput);
        
        if (!Array.isArray(transactions)) {
            throw new Error('Input must be an array of transactions');
        }
        
        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transactions })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        displayBatchResults(result);
    } catch (error) {
        alert('Error analyzing batch: ' + error.message);
        console.error('Batch analysis error:', error);
    }
}

// Display batch results
function displayBatchResults(result) {
    const batchResults = document.getElementById('batchResults');
    const batchSummary = document.getElementById('batchSummary');
    const batchDetails = document.getElementById('batchDetails');
    
    const summary = result.summary;
    
    // Summary section
    batchSummary.innerHTML = `
        <h3>Summary</h3>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="label">Total</div>
                <div class="value">${summary.total_transactions}</div>
            </div>
            <div class="summary-item">
                <div class="label">Fraud</div>
                <div class="value" style="color: var(--danger-color);">${summary.predicted_fraud}</div>
            </div>
            <div class="summary-item">
                <div class="label">Legitimate</div>
                <div class="value" style="color: var(--success-color);">${summary.predicted_legitimate}</div>
            </div>
            <div class="summary-item">
                <div class="label">High Risk</div>
                <div class="value">${summary.high_risk}</div>
            </div>
            <div class="summary-item">
                <div class="label">Medium Risk</div>
                <div class="value">${summary.medium_risk}</div>
            </div>
            <div class="summary-item">
                <div class="label">Low Risk</div>
                <div class="value">${summary.low_risk}</div>
            </div>
        </div>
    `;
    
    // Details section
    const detailsHtml = result.predictions.map((pred, index) => {
        const probability = (pred.fraud_probability * 100).toFixed(2);
        return `
            <div class="transaction-item ${pred.is_fraud ? 'fraud' : ''}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <strong>Transaction #${index + 1}</strong>
                        <div style="margin-top: 5px;">
                            <span class="risk-badge risk-${pred.risk_level}">${pred.risk_level}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2rem; font-weight: 600;">
                            ${pred.is_fraud ? '⚠️ FRAUD' : '✓ Legitimate'}
                        </div>
                        <div style="color: var(--text-secondary);">
                            Probability: ${probability}%
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">
                            Confidence: ${pred.confidence.replace('_', ' ')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    batchDetails.innerHTML = `
        <h3>Transaction Details</h3>
        <div class="transaction-list">
            ${detailsHtml}
        </div>
    `;
    
    batchResults.style.display = 'block';
    batchResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Load model information
async function loadModelInfo() {
    const modelInfo = document.getElementById('modelInfo');
    
    try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        modelInfo.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Model Type</span>
                    <span class="info-value">${data.model_type}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Features Count</span>
                    <span class="info-value">${data.features_count}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Preprocessing</span>
                    <span class="info-value">${data.preprocessing}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Sampling Strategy</span>
                    <span class="info-value">${data.sampling_strategy}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Random State</span>
                    <span class="info-value">${data.random_state}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Status</span>
                    <span class="info-value" style="color: var(--success-color);">✓ ${data.status}</span>
                </div>
            </div>
            
            <div style="margin-top: 25px;">
                <h3>Features (${data.features_count})</h3>
                <div style="max-height: 300px; overflow-y: auto; padding: 15px; background: var(--background); border-radius: 8px; margin-top: 10px;">
                    ${data.features_list.map(feature => `<div style="padding: 5px 0;">${feature}</div>`).join('')}
                </div>
            </div>
        `;
    } catch (error) {
        modelInfo.innerHTML = `
            <div style="text-align: center; color: var(--danger-color);">
                <p>Error loading model information</p>
                <p style="font-size: 0.9rem; margin-top: 10px;">${error.message}</p>
            </div>
        `;
        console.error('Model info error:', error);
    }
}

// Load sample data
function loadSampleData() {
    document.getElementById('step').value = 1;
    document.getElementById('type').value = 'TRANSFER';
    document.getElementById('amount').value = 9000.60;
    document.getElementById('oldbalanceOrg').value = 9000.60;
    document.getElementById('newbalanceOrig').value = 0.00;
    document.getElementById('oldbalanceDest').value = 0.00;
    document.getElementById('newbalanceDest').value = 0.00;
}

// Reset form
function resetForm() {
    document.getElementById('transactionForm').reset();
    document.getElementById('resultsCard').style.display = 'none';
}
