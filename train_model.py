import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def train():
    print("Loading dataset...")
    df = pd.read_csv('bank_transactions_data_2_augmented_clean_2.csv')

    print("Preprocessing data...")
    # Features we want to use
    features = [
        'TransactionAmount', 'CustomerAge', 'TransactionDuration', 
        'LoginAttempts', 'AccountBalance', 'TransactionType', 'Channel'
    ]
    
    # We create a new dataframe with just the features we need to encode
    X = df[features].copy()
    
    # Encode Categoricals using a simple mapping so it's easy to replicate in Flask
    channel_mapping = {'Online': 0, 'Branch': 1, 'ATM': 2}
    type_mapping = {'Debit': 0, 'Credit': 1}
    
    X['Channel'] = X['Channel'].str.strip().map(channel_mapping).fillna(0)
    X['TransactionType'] = X['TransactionType'].str.strip().map(type_mapping).fillna(0)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Isolation Forest model...")
    # Using auto contamination, assuming 5% anomalies
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)
    
    print("Saving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Training Complete. Model and scaler saved.")

if __name__ == '__main__':
    train()
