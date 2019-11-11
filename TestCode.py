# Importing libraries
import pandas as pd
import numpy as np
import datetime
from keras.layers import LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt

# Reading input data
jobData = pd.read_csv('jobData.csv')
anonymizedJob = [jobData]
df = pd.DataFrame(jobData)

# Identifying and handling null values within the data set
nulls = pd.DataFrame(jobData.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Converting categorical features into numerical features
for job in anonymizedJob:
    job['Job Result'] = job['Job Result'].map({'SUCCESS': 0, 'FAILURE': 1, 'CANCELLED': 2}).astype(int)
    #job['Product Name'] = job['Product Name'].map({'Product 1': 0, 'Product 2': 1, 'Product 3': 2, 'Product 4': 3, 'Product 5': 4}).astype(int)
    # job['Component Being updated'] = job['Component Being updated'].map({'Component 1': 0, 'Component 2': 1, 'Component 3': 2, 'Component 4': 3, 'Component 5': 4}).astype(int)
    job['Process'] = job['Process'].map({'Deploy': 0, 'Expand': 1, 'Update': 2}).astype(int)

print(type(job['Job Creation Time']))
# Displaying the data set updated to provide numerical values to non-numerical features
print('Updated Job Dataset displaying values corresponding to Non-numerical features')
print(job)

# Identifying predictors and target variables along with training and test sets
jobArray = job.values
x = jobArray[:, 3:]
y = jobArray[:, 0]

# Dropping features not correlated to the target
x = job.drop(['Customer ID'], axis=1)
print('Predictor features are', x)
print('Target variable is', y)

# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fitting Naive bayes model assuming the data follows a Gaussian distribution
modelNB = GaussianNB()
modelNB_train = modelNB.fit(x_train, y_train)
print('The model fit on the training data set using Naive Bayes approach')
print(modelNB_train, '\n')

# Predicting the results of the model on the test data
y_predictNB = modelNB.predict(x_test)

# Computing the error rate of the model fit
print('Accuracy of the Naive Bayes model fit is', metrics.accuracy_score(y_test, y_predictNB), '\n')
