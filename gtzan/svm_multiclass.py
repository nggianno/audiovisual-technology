from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import array


#load data

x=[]
y=[]
with open('zero_crossings.txt') as f:
   for line in f:
       x.append(line.rstrip())
with open('labels.txt') as f:
   for line in f:
       y.append(line.rstrip())

x = array(x)
x = x.astype('float32')

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
#(x,y) =load_data()


print('Total songs saved: ',len(x))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
print('Songs for training:',len(X_train))
print('Songs for validation:', len(X_test))


 # fit a linear_model.SGDClassifier() model to the data
model = linear_model.SGDClassifier()
model.fit(X_train, y_train)
print(); print(model)

# make predictions
expected_y  = y_test
predicted_y = model.predict(X_test)

# summarize the fit of the model
print(); print(metrics.classification_report(expected_y, predicted_y,  target_names=['classical','metal','pop','reggae','blues','coutry','hiphop','disco','rock','jazz']))
print(); print(metrics.confusion_matrix(expected_y, predicted_y))

# fit a linear_model.PassiveAggressiveClassifier() model to the data
model = linear_model.PassiveAggressiveClassifier()
model.fit(X_train, y_train)
print(); print(model)

# make predictions
expected_y  = y_test
predicted_y = model.predict(X_test)

# summarize the fit of the model
print(); print(metrics.classification_report(expected_y, predicted_y, target_names=['classical','metal','pop','reggae','blues','coutry','hiphop','disco','rock','jazz']))
print(); print(metrics.confusion_matrix(expected_y, predicted_y))

# fit a linear_model.PassiveAggressiveClassifier() model to the data
model = linear_model.RidgeClassifier()
model.fit(X_train, y_train)
print(); print(model)

# make predictions
expected_y  = y_test
predicted_y = model.predict(X_test)

# summarize the fit of the model
print(); print(metrics.classification_report(expected_y, predicted_y, target_names=['classical','metal','pop','reggae','blues','coutry','hiphop','disco','rock','jazz']))
