# Required imports
import os
import pandas as pd
import numpy as np
import kagglehub
import joblib
from collections import Counter

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class FraudDetectionModel:
    """
    A complete fraud detection model with data loading, preprocessing, training, and evaluation.
    Simplified and production-ready implementation.
    """
    
    def __init__(self, random_state=101):
        self.random_state = random_state
        self.model_pipeline = None
        self.feature_columns = None
        
    def load_data(self):
        """Load the PaySim fraud detection dataset from Kaggle."""
        try:
            path = kagglehub.dataset_download("ealaxi/paysim1")
            print(f"Dataset downloaded to: {path}")
            
            csv_file_path = os.path.join(path, 'PS_20174392719_1491204439457_log.csv')
            
            if not os.path.exists(csv_file_path):
                files = os.listdir(path)
                csv_files = [f for f in files if f.endswith('.csv')]

                if csv_files:
                    csv_file_path = os.path.join(path, csv_files[0])
                    print(f"Using file: {csv_files[0]}")

                else:
                    raise FileNotFoundError("No CSV file found in dataset directory.")
            
            df = pd.read_csv(csv_file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def engineer_features(self, df):
        """Create meaningful features for fraud detection."""
        df = df.copy()
        
        df['orig_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        df['orig_error_flag'] = (df['oldbalanceOrg'] - df['amount'] != df['newbalanceOrig']).astype(int)
        
        transfer_cash_mask = df['type'].isin(['TRANSFER', 'CASH_IN'])
        df['dest_error_flag'] = 0

        df.loc[transfer_cash_mask, 'dest_error_flag'] = (
            df.loc[transfer_cash_mask, 'oldbalanceDest'] + df.loc[transfer_cash_mask, 'amount'] 
            != df.loc[transfer_cash_mask, 'newbalanceDest']
        ).astype(int)
        
        df['amount_to_orig_ratio'] = np.where(
            df['oldbalanceOrg'] > 0, 
            df['amount'] / df['oldbalanceOrg'], 
            0
        )
        
        df['amount_to_dest_ratio'] = np.where(
            df['oldbalanceDest'] > 0, 
            df['amount'] / df['oldbalanceDest'], 
            0
        )
        
        return df
    
    def preprocess_data(self, df):
        """Clean and prepare data for modeling."""
        df = df.copy()
        
        columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        df = self.engineer_features(df)
        
        # One-hot encode transaction type
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)
        
        print(f"Data preprocessed. Final shape: {df.shape}")
        return df
    
    def split_data(self, df, test_size=0.25):
        """Split data into training and testing sets."""
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Fraud cases in training: {y_train.sum()} ({y_train.mean():.2%})")
        print(f"Fraud cases in test: {y_test.sum()} ({y_test.mean():.2%})")
        
        return X_train, X_test, y_train, y_test
    
    def create_model_pipeline(self):
        """Create a complete ML pipeline with preprocessing and model."""
        
        numerical_cols = []
        categorical_cols = []
        
        for col in self.feature_columns:
            if col.startswith('type_') or col.endswith('_flag'):
                categorical_cols.append(col)  
            else:
                numerical_cols.append(col)    
        
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        # Define preprocessing for different column types
        preprocessor = ColumnTransformer( 
            transformers=[
                ('scaler', StandardScaler(), numerical_cols),  
                ('passthrough', 'passthrough', categorical_cols) 
            ],
            remainder='drop'  
        )
        
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy=0.1, random_state=self.random_state)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        return pipeline
    
    def train_model(self, X_train, y_train):
        """Train the fraud detection model."""
        print("Training model...")
        
        self.model_pipeline = self.create_model_pipeline()
        self.model_pipeline.fit(X_train, y_train)
        
        print("Model training completed!")
        return self.model_pipeline
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model_pipeline is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model_pipeline.predict(X_test)
        y_pred_proba = self.model_pipeline.predict_proba(X_test)[:, 1]
        
        # Print evaluation metrics
        cm = confusion_matrix(y_test, y_pred)

        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(classification_report(y_test, y_pred))
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC Score: {auc_score:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return {
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def save_model(self, filename='fraud_detection_model.joblib'):
        """Save the trained model."""
        if self.model_pipeline is None:
            raise ValueError("No trained model to save")
        
        joblib.dump(self.model_pipeline, filename)
        print(f"Model saved as: {filename}")
        
    def load_model(self, filename='fraud_detection_model.joblib'):
        """Load a pre-trained model."""
        self.model_pipeline = joblib.load(filename)
        print(f"Model loaded from: {filename}")
        
        
    def predict_fraud(self, transaction_data):
        """Predict fraud for new transaction data."""
        if self.model_pipeline is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        probabilities = self.model_pipeline.predict_proba(transaction_data)[:, 1]
        predictions = self.model_pipeline.predict(transaction_data)
        
        return predictions, probabilities


def main():
    """Main function to run the complete fraud detection pipeline."""
    
    fraud_detector = FraudDetectionModel(random_state=42)
    
    try:
        print("Loading data...")
        df = fraud_detector.load_data()
        
        print("Preprocessing data...")
        df_processed = fraud_detector.preprocess_data(df)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = fraud_detector.split_data(df_processed)
        
        fraud_detector.train_model(X_train, y_train)
        
        results = fraud_detector.evaluate_model(X_test, y_test)
        
        fraud_detector.save_model('fraud_detection_model.joblib')
        
        print("\nFraud detection model pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()