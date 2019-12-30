from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#load data
f = open('zero_crossings.txt', 'r')
x = f.readlines()
f.close()
f = open('labels.txt', 'r')
y = f.readlines()
f.close()
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
#(x,y) =load_data()

print('Total songs saved: ',len(x))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
print('Songs for training:',len(X_train))
print('Songs for validation:', len(X_test))


svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
