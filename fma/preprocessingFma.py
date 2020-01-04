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

def make_tracks_dataset(data):
     clean_tracks = data[['track_id',"genre_top"]] 
     clean_tracks.drop(clean_tracks.index[0], inplace=True)
     # !!!!!!!!!
     # Do you want merge with echonest? Uncomment the next one 
     # clean_tracks = clean_tracks.merge(echonest,on="track_id",how="inner")
     #
     frequencies = clean_tracks['genre_top'].value_counts()
     print(frequencies) 
     # !!!!!!!!!
     # Do you want merge with features? Uncomment the next one 
     # clean_tracks = clean_tracks.merge(features,on="track_id",how="inner")
     #
     data_final = data.merge(features, on='track_id', how='inner')
     frequencies = clean_tracks['genre_top'].value_counts()
     print(frequencies) 
     
     # select genres and how many songs each genre
     data = pd.DataFrame()
     selected_genres = ['Pop','Rock','Hip-Hop','Classical']
     clean_tracks= clean_tracks.sample(frac=1).reset_index(drop=True)
     for i in selected_genres:
          songs = clean_tracks.loc[clean_tracks['genre_top'] == i]
          songs = songs.head(1000)
          data = data.append(songs)
     data.reset_index(inplace=True)
     data.drop(columns='index', inplace=True)
     print("\ndata_final:\n",data_final['genre_top'].value_counts())
     
     # keep only the feautures that we want : column names for both echonest or features 
     # if they are merged
     desired_features='track_id','genre_top',"spectral_bandwidth.2",'zcr.2','spectral_centroid.2', 'spectral_rolloff.2'
     dataset= data_final.loc[:,desired_features]
     
     max_tracks_each_genre = 800
     classical = dataset[dataset['genre_top'] == 'Classical'].head(max_tracks_each_genre)
     rock = dataset[dataset['genre_top'] == 'Rock'].head(max_tracks_each_genre)
     hip_hop = dataset[dataset['genre_top'] == 'Hip-Hop'].head(max_tracks_each_genre)
     pop = dataset[dataset['genre_top'] == 'Pop'].head(max_tracks_each_genre)
     
     # combine each to our final dataset 
     final_dataset = pd.DataFrame()
     final_dataset = final_dataset.append([classical,hip_hop,rock,pop])

     return final_dataset

def extract_to_csv(df):

      df.to_csv(path_or_buf='./data/final.csv')

      return 


if __name__=='__main__':
     #set directory paths
     TRACK_PATH = './data/tracks.csv'
     GENRE_PATH = './data/genres.csv'
     FEATURE_PATH = './data/features.csv'
     ECH0NEST_PATH = './data/echonest.csv'


     #csv to dataframe
     tracks = pd.read_csv(TRACK_PATH,header = 1 )
     genres = pd.read_csv(GENRE_PATH)
     features = pd.read_csv(FEATURE_PATH)
     echonest = pd.read_csv(ECH0NEST_PATH,header = 2)

     (tracks,echonest,features) = preprocess_csv_files(tracks,echonest,features)
     """make a dataset of 400 * 4 categories(POP,ROCK,CLASSICAL,HIP-HOP)"""
   
     final = make_tracks_dataset(tracks)
     
     final.rename(columns={"zcr.2": "zcr"}, inplace=True)
     final.rename(columns={"spectral_rolloff.2": "spectral_rollof"}, inplace=True)
     final.rename(columns={"spectral_centroid.2": "spectral_centroid"}, inplace=True)
     print(final.head())
     extract_to_csv(final)


