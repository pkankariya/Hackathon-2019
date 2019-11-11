# Importing libraries
import pandas as pd
import numpy as np
import datetime
from keras.layers import LSTM
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
dataDeploy = pd.read_csv('Anonymized Deployment Job Data.csv')
dataExpand = pd.read_csv('Anonymized Expansion Job Data.csv')
dataUpdate = pd.read_csv('Anonymized Update Job Data.csv')

anonymizedJob = [jobData]
df = pd.DataFrame(jobData)
print(df)

# Identifying and handling null values within the data set
nulls = pd.DataFrame(jobData.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
jobData.fillna(0)


# Converting categorical features into numerical features
for job in anonymizedJob:
    job['Job Result'] = job['Job Result'].map({'SUCCESS': 0, 'FAILURE': 1, 'CANCELLED': 2}).astype(int)
    # job['Product Name'] = job['Product Name'].map({'Product 1': 0, 'Product 2': 1, 'Product 3': 2, 'Product 4': 3, 'Product 5': 4}).astype(int)
    # job['Component Being updated'] = job['Component Being updated'].map({'Component 1': 0, 'Component 2': 1, 'Component 3': 2, 'Component 4': 3, 'Component 5': 4}).astype(int)
    job['Process'] = job['Process'].map({'Deploy': 0, 'Expand': 1, 'Update': 2}).astype(int)

# # Updating data and time information of the job
# creationTime = job['Job Creation Time']
# createTime = list(map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y  %H:%M').strftime('%m%d%Y %H:%M'), creationTime))
# # createTime = datetime.datetime.strptime(creationTime, "%m/%d/%Y  %H:%M")
# print('The data and time in their appropriate format is displayed as: ',createTime)

# Displaying the data set updated to provide numerical values to non-numerical features
print('Updated Job Dataset displaying values corresponding to Non-numerical features')
print(job)

# Correlation of numerical features associated with target
numeric_features = job.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Features positively correlated to the target:')
print(corr['Job Result'].sort_values(ascending=False)[:5], '\n')
print('Features negatively correlated to the target:')
print(corr['Job Result'].sort_values(ascending=False)[-5:])

# Identifying predictors and target variables along with training and test sets
jobArray = job.values
x = jobArray[:, 1:]
y = jobArray[:, 0]

# Dropping features not correlated to the target
x = job.drop(['Product Name', 'Component Being updated', 'Customer ID', 'Job Creation Time', 'Job Completion Times'], axis=1)
print('Predictor features are', x)
print('Target variable is', y)

# Defining the training and test data sets along with ground truth and predicted labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('Training predictor features')
print(x_train)
print('Test predictor features')
print(x_test)

# Fitting Linear SVM model
svm = SVC()
modelSVM_train = svm.fit(x_train, y_train)
train_score = svm.score(x_train, y_train)
print('The model fit on the training data set using Support Vector Machines approach')
print(modelSVM_train, '\n')

# Predicting the results of the model on the test data
y_predictSVM = svm.predict(x_test)

# Computing the error rate of the model fit
accuracy_svm = round(svm.score(x_train, y_train) * 100, 2)
print('Accuracy of the linear SVM model fit is', accuracy_svm, '\n')