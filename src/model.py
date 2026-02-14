import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    # Load Data
    data_path = 'data/telecom_churn.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run src/data_generator.py first.")
        return

    df = pd.read_csv(data_path)

    # Preprocessing
    # Drop CustomerID as it's not a feature
    df = df.drop('customerID', axis=1)

    # Handle TotalCharges (convert to numeric, coerce errors)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Encode Categorical Variables
    # We will save the encoders to use them in the app
    encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split Data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Numerical Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training Random Forest Model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save Artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    
    # Save feature names for potential use
    joblib.dump(X.columns.tolist(), 'models/features.pkl')
    
    print("Model and artifacts saved to models/")

if __name__ == "__main__":
    train_model()
