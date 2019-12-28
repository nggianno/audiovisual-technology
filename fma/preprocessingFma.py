import os
import IPython.display as ipd
import numpy as np
import pandas as pd
# Load metadata and features.
tracks = pd.read_csv('tracks.csv',header = 1 )
genres = pd.read_csv('genres.csv')
features = pd.read_csv('features.csv')
echonest = pd.read_csv('echonest.csv',header = 2 )

tracks.rename(columns={ tracks.columns[0]: "track_id" }, inplace = True)
echonest.rename(columns={ echonest.columns[0]: "track_id" }, inplace = True)
features.rename(columns={ features.columns[0]: "track_id" }, inplace = True)

features.head(10)
tracks.head(12)
genres.head(12)
features.columns
tracks.columns
# make a new df keeping only this columns and delete the first row 
# index 0 was : tracks id NAN NAN NAN 

clean_tracks = tracks[['track_id',"bit_rate","genre_top","genres"]]
# clear workspace 
del tracks
####
clean_tracks.drop( clean_tracks.index[0] , inplace = True ) 
clean_tracks.head(10)
# clean_na_tracks : tracks with no NAN in column "genre_top"
clean_na_tracks = clean_tracks.dropna()
# songs of each genre
frequencies     = clean_na_tracks['genre_top'].value_counts()
frequencies
