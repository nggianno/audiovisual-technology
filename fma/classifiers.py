import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier



def load_data(path):
    data = pd.read_csv(path)
    data.drop('Unnamed: 0',axis=1,inplace=True)
    #for gtzan
    # data.drop('',axis=1,inplace=True)

    return data

def naive_bayes_classifier(X_train,y_train,X_test):

    #Create a Gaussian Classifier
    gnb = GaussianNB()
    #Train the model using the training sets
    gnb.fit(X_train, y_train)
    #Predict the response for test dataset
    prediction = gnb.predict(X_test)

    return prediction

def logistic_regression(X_train,y_train,X_test):

    clf = linear_model.LogisticRegression(C=1.0)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def svm_classifier(X_train,y_train,X_test):

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    # make predictions
    prediction = svclassifier.predict(X_test)

    return prediction

def knn_classifier(X_train,y_train,X_test):

    knn = KNeighborsClassifier(weights="uniform")
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)

    return prediction

def random_forest(X_train,y_train,X_test):
    clf = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=42)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def adaboost(X_train,y_train,X_test):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def gradient_boost(X_train,y_train,X_test):
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def plot_cm(y_test,y_pred,method):

    cm = metrics.confusion_matrix(y_test, y_pred, labels=['Pop','Rock','Hip-Hop','Classical'])
    print(metrics.classification_report(y_test,y_pred,labels=['Pop','Rock','Hip-Hop','Classical']))
    print(cm)
    cm_df = pd.DataFrame(cm,index=['Pop','Rock','Hip-Hop','Classical'],
                     columns=['Pop','Rock','Hip-Hop','Classical'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(cm_df, annot=True,linewidths=.2,fmt="d")
    plt.title('{0} | Accuracy:{1:.3f}'.format(method, metrics.accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.get_ylim()
    ax.set_ylim(4.4, 0) # !adjust the y limit to look better
    plt.show()

    return

if __name__ == '__main__':

    #labels = ['Hip-Hop', 'Rock', 'Electronic', 'Classical']---final6
    #labels = ['Pop','Rock','Hip-Hop','Classical'] ---final4
    #labels = ['Pop', 'Rock', 'Electronic', 'Classical'] ---final5
    #labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']---gtzan

    DATA_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final3.csv'
    GTZAN_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/gtzan-metadata.csv'
    TEST_SIZE = 0.2
    # read dataset
    dataset = load_data(DATA_PATH)
    print(dataset.head(10))
    print(dataset.shape)
    # pass input columns to X & target column to y
    y = dataset['genre_top']
    X = dataset.drop('genre_top', axis=1)
    #y = dataset['label']
    #X = dataset.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    y_pred1 = naive_bayes_classifier(X_train,y_train,X_test)
    y_pred2 = knn_classifier(X_train,y_train,X_test)
    y_pred3 = random_forest(X_train,y_train,X_test)
    y_pred4 = adaboost(X_train,y_train,X_test)
    y_pred5 = gradient_boost(X_train,y_train,X_test)
    #y_pred6 = svm_classifier(X_train,y_train,X_test)
    y_pred7 = logistic_regression(X_train,y_train,X_test)

    plot_cm(y_test,y_pred1,method = 'Naive-Bayes')
    plot_cm(y_test,y_pred2,method = 'KNN')
    plot_cm(y_test,y_pred3,method = 'RandomForest')
    plot_cm(y_test,y_pred4,method = 'Adaboost')
    plot_cm(y_test,y_pred5,method = 'GradientBoost')
    #plot_cm(y_test,y_pred6,method = 'SVM')
    plot_cm(y_test,y_pred7,method='Logistic Regression')
