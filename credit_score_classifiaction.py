import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    'Age': [25, 35, 45, 30, 20, 50, 40, 28, 32, 48],
    'Income': [50000, 80000, 120000, 60000, 40000, 150000, 100000, 55000, 70000, 130000],
    'Loan_Amount': [10000, 15000, 20000, 12000, 8000, 25000, 18000, 11000, 14000, 22000],
    'Credit_Score': ['Good', 'Good', 'Good', 'Good', 'Bad', 'Good', 'Good', 'Good', 'Bad', 'Good']
}
df = pd.DataFrame(data)
label_encoder = LabelEncoder()
df['Credit_Score'] = label_encoder.fit_transform(df['Credit_Score'])  # Encode 'Good' as 1 and 'Bad' as 0
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))