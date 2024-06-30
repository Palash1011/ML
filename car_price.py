import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'age': [1, 3, 5, 2, 7, 4, 3, 5, 2, 9],
    'mileage': [15000, 30000, 50000, 20000, 70000, 40000, 30000, 50000, 20000, 90000],
    'make': ['Toyota', 'Ford', 'BMW', 'Toyota', 'Ford', 'BMW', 'Toyota', 'Ford', 'BMW', 'Toyota'],
    'model': ['Corolla', 'Focus', 'X3', 'Camry', 'Fiesta', 'X5', 'Corolla', 'Focus', 'X3', 'Camry'],
    'price': [22000, 18000, 35000, 24000, 16000, 33000, 21000, 17500, 34000, 20000]
}
df = pd.DataFrame(data)
label_encoder_make = LabelEncoder()
label_encoder_model = LabelEncoder()
df['make'] = label_encoder_make.fit_transform(df['make'])
df['model'] = label_encoder_model.fit_transform(df['model'])
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
pred_vs_actual = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(pred_vs_actual)