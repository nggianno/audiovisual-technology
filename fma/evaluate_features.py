import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from preprocessingFma import preprocess_csv_files
from preprocessingFma import make_tracks_dataset
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import feature_selection
from sklearn.decomposition import PCA



def ensemblers(input,target):


    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, input, target, cv=10)
    print("Cross-validation score - Decision Trees:\n{}".format(scores.mean()))

    clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y, cv=20)
    print("Cross-validation score - Extra Trees:\n{}".format(scores.mean()))

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, eclf],
                          ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    return

def feature_selection_metrics(input,target):

    mutual_info = feature_selection.mutual_info_classif(input,target)
    print(mutual_info)
    plt.bar(range(len(input.columns)), mutual_info)
    plt.xticks(range(len(input.columns)), input.columns, rotation='vertical')
    plt.figure(1)
    plt.title('Mutual information for each feature')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information')
    plt.show()

    selector = feature_selection.SelectKBest(k=4)
    selector.fit(input,target)
    scores = -np.log10(selector.pvalues_)
    plt.figure(2)
    plt.bar(range(len(input.columns)), scores)
    plt.xticks(range(len(input.columns)), input.columns, rotation='vertical')
    plt.title('10 best feature selection barplot')
    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.show()

    # model = LogisticRegression()
    # rfe = RFE(model, 4)
    # fit = rfe.fit(input, target)
    # print(input.columns)
    # truth_table = fit.support_
    # #input.columns[truth_table == True]
    # best_features = np.array(input.columns[truth_table == True])
    # cleaned_dataset = input[best_features]

    model = ExtraTreesClassifier()
    model.fit(input, target)
    print("Feature importances using ExtraTreesClassifier:\n{}".format(model.feature_importances_))
    # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=input.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature importance method')
    plt.show()

    pca = PCA(n_components=10)
    fit = pca.fit(input)
    # summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)


if __name__ == '__main__':
    # csv to dataframe

    data = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final2.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)

    print(data.head(10))

    #extract_to_csv(data)
    y = data['genre_top']
    X = data.drop('genre_top', axis=1)


    #ensemblers(X,y)
    feature_selection_metrics(X,y)

    """after running feature_selection_metrics we see that tonnetz gives weak info so we drop the respective columns"""
    data.drop(labels=data.ix[:, 'tonnetz.12':'tonnetz.17'].columns, axis=1, inplace=True)
    print(data.columns)

    #data.to_csv(path_or_buf='/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final3.csv')



