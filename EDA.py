# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv('Telco-Customer-Churn.csv')

# ------------------ 1. Basic Info ------------------
print("Shape of dataset:", df.shape)
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())

# ------------------ 2. Target Variable Distribution ------------------
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Churn Distribution')
plt.show()

print("\nChurn Percentage:")
print(df['Churn'].value_counts(normalize=True)*100)

# ------------------ 3. Handle Missing Values ------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nMissing values after conversion:", df['TotalCharges'].isnull().sum())
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# ------------------ 4. Numerical Features Analysis ------------------
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\nNumerical Features Summary:")
print(df[num_features].describe())

# Histograms
df[num_features].hist(bins=30, figsize=(10,6), color='skyblue')
plt.suptitle("Distribution of Numerical Features")
plt.show()

# ------------------ 5. Categorical Features vs Churn ------------------
cat_features = ['gender','SeniorCitizen','Partner','Dependents','PhoneService',
                'MultipleLines','InternetService','OnlineSecurity','TechSupport',
                'Contract','PaperlessBilling','PaymentMethod']

for col in cat_features:
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x=col, hue='Churn', palette='Set2')
    plt.title(f'{col} vs Churn')
    plt.xticks(rotation=45)
    plt.show()

# ------------------ 6. Correlation Heatmap ------------------
df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ------------------ 7. Tenure vs Churn ------------------
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title('Tenure vs Churn')
plt.show()
