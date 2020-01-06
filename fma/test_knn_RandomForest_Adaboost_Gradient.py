import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# read a csv from a directory named data in 
# current directory 
data =  pd.read_csv("./data/final.csv",)
data = data.drop("Unnamed: 0",axis =1 )
data = data.sample(frac=1).reset_index(drop=True)

#tracks_final = tracks.merge(features, on='track_id', how='inner')
data.columns
# if you choose 6:len(data.columns) in the next line 
# you can take only mfccs
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:len(data.columns)],data.loc[:,"genre_top"], test_size=0.2)

#----------knn classification---------------
knn = KNeighborsClassifier(weights = "uniform")
knn
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

#-----------plots---------------------------
labels = ["Classical","Rock","Hip-Hop","Pop"]
cm = confusion_matrix(y_test,y_pred, labels)

cm_df = pd.DataFrame(cm,
                     index = ['Classical','Rock','Hip-Hop',"Pop"], 
                     columns = ['Classical','Rock','Hip-Hop',"Pop"])
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(cm_df, annot=True,linewidths=.2)
plt.title('Knn \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
ax.get_ylim()
ax.set_ylim(4.4, 0) # !adjust the y limit to look better
plt.show()

clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
#----------------------------------------------------
labels = ["Classical","Rock","Hip-Hop","Pop"]
cm = confusion_matrix(y_test,y_pred, labels)

cm_df = pd.DataFrame(cm,
                     index = ['Classical','Rock','Hip-Hop',"Pop"], 
                     columns = ['Classical','Rock','Hip-Hop',"Pop"])
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(cm_df, annot=True,linewidths=.2)
plt.title('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
ax.get_ylim()
ax.set_ylim(4.4, 0) # !adjust the y limit to look better
plt.show()

#----------------------------------------------------
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)


labels = ["Classical","Rock","Hip-Hop","Pop"]
cm = confusion_matrix(y_test,y_pred, labels)

cm_df = pd.DataFrame(cm,
                     index = ['Classical','Rock','Hip-Hop',"Pop"], 
                     columns = ['Classical','Rock','Hip-Hop',"Pop"])
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(cm_df, annot=True,linewidths=.2)
plt.title('Adaboost \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
ax.get_ylim()
ax.set_ylim(4.4, 0) # !adjust the y limit to look better
plt.show()

#----------------------------------------
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

labels = ["Classical","Rock","Hip-Hop","Pop"]
cm = confusion_matrix(y_test,y_pred, labels)

cm_df = pd.DataFrame(cm,
                     index = ['Classical','Rock','Hip-Hop',"Pop"], 
                     columns = ['Classical','Rock','Hip-Hop',"Pop"])
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(cm_df, annot=True,linewidths=.2)
plt.title('Gradient Boost \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
ax.get_ylim()
ax.set_ylim(4.4, 0) # !adjust the y limit to look better
plt.show()

