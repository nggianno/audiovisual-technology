import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#read dataset
data = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
#print(data.head(10))
print(data[['zcr','spectral_centroid','spectral_rollof']])
#split train test
X_train, X_test, y_train, y_test = train_test_split(data[['zcr','spectral_centroid','spectral_rollof']], data['genre_top'], test_size=0.2)
#Create a Gaussian Classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",metrics.multilabel_confusion_matrix(y_test,y_pred,labels=['Pop','Rock','Hip-Hop','Classical']))
print(metrics.classification_report(y_test,y_pred))