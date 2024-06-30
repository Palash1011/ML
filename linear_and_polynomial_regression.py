import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(42)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)
X = X[:, np.newaxis]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_pred_poly = poly_reg.predict(X_poly_test)
plt.scatter(X, y, s=10)
plt.plot(X_test, y_pred_lin, color='r', label="Linear Regression")
X_grid = np.arange(min(X), max(X), 0.1)[:, np.newaxis]
plt.plot(X_grid, poly_reg.predict(poly.transform(X_grid)), color='b', label="Polynomial Regression (degree 2)")
plt.legend()
plt.show()
print("Linear Regression")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred_lin):.2f}")
print(f"R^2 score: {r2_score(y_test, y_pred_lin):.2f}")
print("\nPolynomial Regression (degree 2)")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R^2 score: {r2_score(y_test, y_pred_poly):.2f}")