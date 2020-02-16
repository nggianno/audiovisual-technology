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
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ClassificationReport
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA





def load_data(path):
    data = pd.read_csv(path)
    data.drop('Unnamed: 0',axis=1,inplace=True)
    #for gtzan
    data.drop('track_id',axis=1,inplace=True)

    return data

def pca_analysis(X_train,X_test,num):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA(n_components=num)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # summarize components
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance: %s" % explained_variance)
    # print(fit.components_)
    print(pca.n_components_)

    return X_train,X_test

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
    clf = RandomForestClassifier(n_estimators=150, max_features='auto', random_state=42)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def adaboost(X_train,y_train,X_test):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100,learning_rate=0.1)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def gradient_boost(X_train,y_train,X_test):
    clf = GradientBoostingClassifier(n_estimators=120,learning_rate=0.1,max_depth=5)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction

def classification_report(X_train,y_train,X_test,y_test):
    viz = ClassificationReport(GradientBoostingClassifier(n_estimators=120,learning_rate=0.1,max_depth=5),cmap='PuOr')
    viz.fit(X_train,y_train)
    viz.score(X_test,y_test)
    viz.show()
    viz = ClassificationReport(RandomForestClassifier(n_estimators=150, max_features='auto', random_state=42),
                               cmap='PuOr')
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()

    return

def plot_cm(y_test,y_pred,method):
    # labels = ['Hip-Hop', 'Rock', 'Electronic', 'Classical']---final6
    # labels = ['Pop','Rock','Hip-Hop','Classical'] ---final4
    # labels = ['Pop', 'Rock', 'Electronic', 'Classical'] ---final5
    # labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']---gtzan
    # labels = ['Hip-Hop', 'Rock', 'Classical', 'Folk']--final8
    cm = metrics.confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    print(metrics.classification_report(y_test,y_pred,labels=np.unique(y_test)))
    print(cm)
    cm_df = pd.DataFrame(cm,index=np.unique(y_test),
                     columns=np.unique(y_test))
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

    """ Run classifiers on created FMA datasets containing 4 or 8 genres and get the respective results"""

    #labels = ['Hip-Hop', 'Rock', 'Electronic', 'Classical']---final6
    #labels = ['Pop','Rock','Hip-Hop','Classical'] ---final4
    #labels = ['Pop', 'Rock', 'Electronic', 'Classical'] ---final5
    #labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']---gtzan

    DATA_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final6.csv'
    GTZAN_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/gtzan-metadata.csv'
    TEST_SIZE = 0.2
    # read dataset
    dataset = load_data(DATA_PATH)
    print(dataset.head(10))
    print(dataset.shape)

    dataset.drop(labels=dataset.ix[:, 'tonnetz.12':'tonnetz.17'].columns, axis=1, inplace=True)
    dataset.drop(labels=dataset.ix[:, 'chroma_cens.24':'chroma_cens.35'].columns, axis=1, inplace=True)
    print(dataset.head(10))
    print(dataset.shape)
    # pass input columns to X & target column to y
    y = dataset['genre_top']
    X = dataset.drop('genre_top', axis=1)
    #y = dataset['label']
    #X = dataset.drop('label', axis=1)

    """split the data to training and testing"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    #apply PCA
    PCA_VECTORS = 30
    #X_train,X_test = pca_analysis(X_train, X_test, PCA_VECTORS)

    """run classification report function to visualize Recall / Precision / F1-Score for each music genre"""
    classification_report(X_train, y_train, X_test, y_test)


    y_pred1 = naive_bayes_classifier(X_train,y_train,X_test)
    y_pred2 = knn_classifier(X_train,y_train,X_test)
    y_pred3 = random_forest(X_train,y_train,X_test)
    y_pred4 = adaboost(X_train,y_train,X_test)
    y_pred5 = gradient_boost(X_train,y_train,X_test)
    y_pred6 = svm_classifier(X_train,y_train,X_test)
    y_pred7 = logistic_regression(X_train,y_train,X_test)

    plot_cm(y_test,y_pred1,method = 'Naive-Bayes')
    plot_cm(y_test,y_pred2,method = 'KNN')
    plot_cm(y_test,y_pred3,method = 'RandomForest')
    plot_cm(y_test,y_pred4,method = 'Adaboost')
    plot_cm(y_test,y_pred5,method = 'GradientBoost')
    plot_cm(y_test,y_pred6,method = 'SVM')
    plot_cm(y_test,y_pred7,method='Logistic Regression')
