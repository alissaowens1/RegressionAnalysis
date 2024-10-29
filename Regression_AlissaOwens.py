# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:37:48 2023

@author: alissa owens
Apply at least two regression models in one of the datasets from 
https://archive.ics.uci.edu/ml/datasets.php
Compare between these two models using  R^2 and Root Mean Squared Error.
(use a bar graph for comparison)

"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

# Load dataset
data = pd.read_csv('iris.data')

# Split data into predictors and target
X = data.drop('Iris-setosa', axis=1)
y = data['3.5']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
linear_regression = LinearRegression()

# Train the model
linear_regression.fit(X_train, y_train)

# Make predictions
y_pred = linear_regression.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


from sklearn.linear_model import LogisticRegression

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logistic_regression = LogisticRegression()

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Train the model
logistic_regression.fit(X_train, y_train)

# Make predictions
y_pred = logistic_regression.predict(X_test)

# Evaluate the model
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Make predictions
y_pred = linear_regression.predict(X_test)

# Calculate R^2
r2_linear = r2_score(y_test, y_pred)
print('R^2:', r2_linear)

# Calculate RMSE
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse_linear)

# Make predictions
y_pred = logistic_regression.predict(X_test)

# Calculate R^2
r2_logistic = r2_score(y_test, y_pred)
print('R^2:', r2_logistic)

# Calculate RMSE
rmse_logistic = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse_logistic)

import matplotlib.pyplot as plt

r2_scores = [r2_linear, r2_logistic]
rmse_scores = [rmse_linear, rmse_logistic]

labels = ['Linear Regression', 'Logistic Regression']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, r2_scores, width, label='R^2')
rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE')


ax.set_ylabel('Scores')
ax.set_title('Scores by model and metric')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
