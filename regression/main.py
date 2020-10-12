# Import libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Import data
data = pd.read_csv("data.csv")
print(data)

# Divide the dataset
X = data.iloc[:, 1:2].values  # Temperature values
y = data.iloc[:, 2].values  # Pressure values

# Create a linear model and fit!
linear_model = LinearRegression()
linear_model.fit(X, y)

# Graph the linear model
plt.scatter(X, y, color="blue")
plt.plot(X, linear_model.predict(X), color="red")
plt.title("Linear Model")
plt.xlabel("temperature")
plt.ylabel("pressure")
plt.savefig("linear_model.png")

# Create a 4th degree polynomial model and fit!
polynomial_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
polynomial_model.fit(X, y)

# Graph the polynomial model
plt.cla()  # Clear the old plot
plt.scatter(X, y, color="blue")
X = np.array([i / 10.0 for i in range(1000)]).reshape(-1, 1)  # 0.0, 0.1, 0.2, ... 100.0
plt.plot(X, polynomial_model.predict(X), color="red")
plt.title("Polynomial Model")
plt.xlabel("temperature")
plt.ylabel("pressure")
plt.savefig("polynomial_model.png")

# Some predictions
print("Linear model's prediction when temp=50:", linear_model.predict([[50]]))
print("Polynomial model's prediction when temp=50:", polynomial_model.predict([[50]]))