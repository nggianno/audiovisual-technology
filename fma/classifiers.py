import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



def load_data(path):
    data = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final.csv')
    data.drop('Unnamed: 0',axis=1,inplace=True)

    return data

def naive_bayes_classifier(X_train,y_train,X_test):

    #Create a Gaussian Classifier
    gnb = GaussianNB()
    #Train the model using the training sets
    gnb.fit(X_train, y_train)
    #Predict the response for test dataset
    prediction = gnb.predict(X_test)

    return prediction

def svm_classifier(X_train,y_train,X_test):

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    # make predictions
    prediction = svclassifier.predict(X_test)

    return prediction

def knn_classifier(X_train,y_train,X_test):

    knn = KNeighborsClassifier(weights="uniform")
    knn
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)

    return prediction

def plot_cm(y_test,y_pred):

    cm = metrics.confusion_matrix(y_test, y_pred, labels=['Pop','Rock','Hip-Hop','Classical'])

    cm_df = pd.DataFrame(cm,index=['Classical','Rock','Hip-Hop',"Pop"],
                     columns=['Classical','Rock','Hip-Hop',"Pop"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(cm_df, annot=True,linewidths=.2)
    plt.title('Accuracy:{0:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.get_ylim()
    ax.set_ylim(4.4, 0) # !adjust the y limit to look better
    plt.show()

    return

if __name__ == '__main__':

    DATA_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final.csv'
    TEST_SIZE = 0.2
    # read dataset
    dataset = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[['zcr', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rollof']], dataset['genre_top'],
        test_size=TEST_SIZE)
    #Naive-Bayes
    y_pred1 = naive_bayes_classifier(X_train,y_train,X_test)
    #KNN
    y_pred2 = knn_classifier(X_train,y_train,X_test)
    #SVM
    y_pred3 = svm_classifier(X_train, y_train, X_test)

    plot_cm(y_test,y_pred1)
    plot_cm(y_test,y_pred2)
    plot_cm(y_test,y_pred3)