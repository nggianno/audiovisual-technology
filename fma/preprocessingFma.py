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
frequencies # this is Series.obj : arrays with strings ,numbers etc



#*********** make the dataset *******************
my_genres = pd.array(frequencies.index.values)

# we select only 100 songs from each genre 
# so we will exclude the last one from freq ( < 100 songs )
tracks = pd.DataFrame()

for x in my_genres[:15]: 
    songs = clean_na_tracks.loc[clean_na_tracks['genre_top'] == x]
    # keep the first 100 songs 
    songs = songs.head(100)
    tracks = tracks.append(songs)


# our dataset is " tracks  " 
# each genre has 100 songs so 15 genres x 100 songs = 1500




