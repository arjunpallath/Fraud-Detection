# -*- coding: utf-8 -*-
"""bank dataset with Fraud Detection (ML)"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
bank_dataset = pd.read_csv('bank_transactions_data_2_augmented_clean_2.csv')

# Basic info
print(bank_dataset.shape)
print(bank_dataset.head())
print(bank_dataset.info())
print(bank_dataset.isnull().sum())
print(bank_dataset.describe())

# EDA
sns.set()
plt.figure(figsize=(6,6))
sns.histplot(bank_dataset['CustomerAge'], kde=True)
plt.title('Customer Age')
plt.show()

plt.figure(figsize=(6,6))
sns.histplot(bank_dataset['TransactionAmount'], kde=True)
plt.title('Transaction Amount')
plt.show()

print(bank_dataset['Channel'].value_counts())
print(bank_dataset['Location'].value_counts())

# Encode categorical columns
bank_dataset.replace({'Channel':{'Online':0,'Branch':1,'ATM':2}}, inplace=True)
bank_dataset.replace({'TransactionType':{'Debit':0,'Credit':1}}, inplace=True)

# -----------------------------
# Rule-Based Fraud Detection
# -----------------------------
bank_dataset["IsFake"] = ((bank_dataset["TransactionAmount"] > bank_dataset["AccountBalance"]) |
                         (bank_dataset["LoginAttempts"] > 3)).astype(int)

print(bank_dataset[["TransactionAmount","AccountBalance","LoginAttempts","IsFake"]].head())
print(bank_dataset["IsFake"].value_counts())

# -----------------------------
# Machine Learning Fraud Model
# -----------------------------
# Features and target
X = bank_dataset.drop(columns=['IsFake'])
y = bank_dataset['IsFake']

# Drop unnecessary columns
X = X.drop(columns=[
    'TransactionID','AccountID','TransactionDate',
    'Location','DeviceID','MerchantID',
    'CustomerOccupation','IP Address'
], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# User Input Prediction
# -----------------------------
print("\n--- Fraud Prediction System ---")

TransactionAmount = float(input('Enter TransactionAmount: '))
TransactionType = int(input('Enter TransactionType (Debit: 0, Credit: 1): '))
Channel = int(input('Enter Channel (Online: 0, Branch: 1, ATM: 2): '))
CustomerAge = float(input('Enter CustomerAge: '))
TransactionDuration = float(input('Enter TransactionDuration: '))
LoginAttempts = int(input('Enter LoginAttempts: '))
AccountBalance = float(input('Enter AccountBalance: '))

input_data = np.array([[TransactionAmount, TransactionType, Channel,
                        CustomerAge, TransactionDuration,
                        LoginAttempts, AccountBalance]])

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("⚠️ Fraudulent Transaction")
else:
    print("✅ Legitimate Transaction")
