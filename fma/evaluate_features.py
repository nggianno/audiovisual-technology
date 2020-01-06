import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from preprocessingFma import preprocess_csv_files
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


def make_dataset(features,tracks):

    listft = features.loc[0,:]=='mean'
    listft['track_id']=True
    listft = features.loc[0,listft]
    desired_feat=features.loc[:,listft.index]
    desired_feat=desired_feat.loc[3:,:]
    desired_feat=desired_feat.drop(desired_feat.loc[:,'chroma_cens.24':'chroma_stft.35'],axis=1)


    clean_tracks = tracks[['track_id', "genre_top"]]
    clean_tracks.drop(clean_tracks.index[0], inplace=True)

    # Do you want merge with features? Uncomment the next line
    clean_tracks = clean_tracks.merge(desired_feat, on="track_id", how="inner")
    # Here we have at least 1000 songs "Classical" if you choose to merge
    frequencies = clean_tracks['genre_top'].value_counts()
    print("1st step :\n", frequencies)
    # Do you want merge with echonest? Uncomment the next line
    # clean_tracks = clean_tracks.merge(echonest,on="track_id",how="inner")
    #frequencies = clean_tracks['genre_top'].value_counts()
    #print("2nd step :\n", frequencies)

    clean_tracks = clean_tracks.sample(frac=1).reset_index(drop=True)

    #  selected_genres :
    # 'Pop','Rock','Hip-Hop','Classical'
    max_tracks_each_genre = 1000  # choose how many songs you will
    classical = clean_tracks[clean_tracks['genre_top'] == 'Classical'].head(max_tracks_each_genre)
    rock = clean_tracks[clean_tracks['genre_top'] == 'Rock'].head(max_tracks_each_genre)
    hip_hop = clean_tracks[clean_tracks['genre_top'] == 'Hip-Hop'].head(max_tracks_each_genre)
    pop = clean_tracks[clean_tracks['genre_top'] == 'Pop'].head(max_tracks_each_genre)

    # append each to our final dataset
    final_dataset = pd.DataFrame()
    final_dataset = final_dataset.append([classical, hip_hop, rock, pop])
    final_dataset = final_dataset.drop('track_id', axis=1)

    return  final_dataset

def feature_selection(input,target):
    model = ExtraTreesClassifier()
    model.fit(input, target)
    print("Feature importances using ExtraTreesClassifier:\n{}".format(model.feature_importances_))
    # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

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


def extract_to_csv(df):

    df.to_csv(path_or_buf='/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final2.csv')

    return

if __name__ == '__main__':
    # csv to dataframe
    TRACK_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/tracks.csv'
    FEATURE_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/features.csv'
    ECHONEST_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/echonest.csv'

    tracks = pd.read_csv(TRACK_PATH, header=1)
    features = pd.read_csv(FEATURE_PATH)
    echonest = pd.read_csv(ECHONEST_PATH, header=2)

    (tracks, echonest, features) = preprocess_csv_files(tracks, echonest, features)

    data = make_dataset(features,tracks)
    print(data.head(10))

    extract_to_csv(data)
    # pass input columns to X & target column to y
    y = data['genre_top']
    X = data.drop('genre_top', axis=1)

    feature_selection(X,y)



