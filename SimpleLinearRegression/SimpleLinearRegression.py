# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
b0 = regressor.intercept_
b1 = regressor.coef_

print(b0)
print(b1)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
print(y_test)
# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()