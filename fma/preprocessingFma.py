import os
import IPython.display as ipd
import numpy as np
import pandas as pd
# Load metadata and features.

def preprocess_csv_files(tracks,echonest,features):
     tracks.rename(columns={tracks.columns[0]: "track_id"}, inplace=True)
     echonest.rename(columns={echonest.columns[0]: "track_id"}, inplace=True)
     features.rename(columns={features.columns[0]: "track_id"}, inplace=True)

     return tracks,echonest,features

def make_tracks_dataset(tracks):
     clean_tracks = tracks[['track_id', "bit_rate", "genre_top"]]
     # clear workspace

     del tracks
     clean_tracks.drop(clean_tracks.index[0], inplace=True)
     print(clean_tracks.shape)
     # clean_na_tracks : tracks with no NAN in column "genre_top"
     #clean_na_tracks = clean_tracks.dropna()
     # songs of each genre
     frequencies = clean_tracks['genre_top'].value_counts()
     print(frequencies)  # this is Series.obj : arrays with strings ,numbers etc

     my_genres = pd.array(frequencies.index.values)
     #print(my_genres)

     # we select only 100 songs from each genre
     # so we will exclude the last one from freq ( < 100 songs )
     tracks = pd.DataFrame()

     selected_genres = ['Pop','Rock','Hip-Hop','Classical']
     for i in selected_genres:
          songs = clean_tracks.loc[clean_tracks['genre_top'] == i]
          # keep the first 1000 songs
          songs = songs.head(1000)
          tracks = tracks.append(songs)

     tracks.reset_index(inplace=True)
     tracks.drop(columns='index', inplace=True)
     tracks_final = tracks.merge(features, on='track_id', how='inner')
     tracks = pd.DataFrame(tracks_final)
     dataset = tracks[['track_id', 'genre_top', 'zcr.2','rmse.2','spectral_centroid.2','spectral_bandwidth.2','spectral_rolloff.2']]
     classical = dataset[dataset['genre_top'] == 'Classical'].head(400)
     rock = dataset[dataset['genre_top'] == 'Rock'].head(400)
     hip_hop = dataset[dataset['genre_top'] == 'Hip-Hop'].head(400)
     pop = dataset[dataset['genre_top'] == 'Pop'].head(400)

     final_dataset = pd.DataFrame()
     final_dataset = final_dataset.append(classical)
     final_dataset = final_dataset.append(hip_hop)
     final_dataset = final_dataset.append(pop)
     final_dataset = final_dataset.append(rock)
     final_dataset.reset_index(inplace=True)
     final_dataset.drop(columns='index', inplace=True)

     return final_dataset

def extract_to_csv(df):

      df.to_csv(path_or_buf='/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/final.csv')

      return


if __name__=='__main__':
     #set directory paths
     TRACK_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/tracks.csv'
     GENRE_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/genres.csv'
     FEATURE_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/features.csv'
     ECH0NEST_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/echonest.csv'


     #csv to dataframe
     tracks = pd.read_csv(TRACK_PATH,header = 1 )
     genres = pd.read_csv(GENRE_PATH)
     features = pd.read_csv(FEATURE_PATH)
     echonest = pd.read_csv(ECH0NEST_PATH,header = 2)

     print(features.columns)
     (tracks,echonest,features) = preprocess_csv_files(tracks,echonest,features)
     """make a dataset of 400 * 4 categories(POP,ROCK,CLASSICAL,HIP-HOP)"""

     final = make_tracks_dataset(tracks)
     
     final.rename(columns={"zcr.2": "zcr"}, inplace=True)
     final.rename(columns={"rmse.2": "rmse"}, inplace=True)
     final.rename(columns={"spectral_rolloff.2": "spectral_rollof"}, inplace=True)
     final.rename(columns={"spectral_centroid.2": "spectral_centroid"}, inplace=True)
     final.rename(columns={"spectral_bandwidth.2": "spectral_bandwidth"}, inplace=True)

     print(final.head())

     #extract_to_csv(final)







