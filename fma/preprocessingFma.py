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
	 
	 # Do you want merge with features? Uncomment the next line
     clean_tracks = clean_tracks.merge(features,on="track_id",how="inner")
     # Here we have at least 1000 songs "Classical" if you choose to merge
     frequencies = clean_tracks['genre_top'].value_counts()
     print("1st step :\n",frequencies) 
     # 
     # Do you want merge with echonest? Uncomment the next line 
     # clean_tracks = clean_tracks.merge(echonest,on="track_id",how="inner")
     # 
     frequencies = clean_tracks['genre_top'].value_counts()
     print("2nd step :\n",frequencies) 
     
     clean_tracks= clean_tracks.sample(frac=1).reset_index(drop=True)
     # desired features can be both from features df and echonest df
     # if you choose to merge
     desired_features='track_id','genre_top',"spectral_bandwidth.2",'zcr.2','spectral_centroid.2','spectral_rolloff.2'
     dataset= clean_tracks.loc[:,desired_features]
     # Uncomment the next line if you choose to merge
     dataset = dataset.join(clean_tracks[c[36:56]]) 
	 #!!!!!!!!!! here index 36:56 are the mffc mean in features
     
     #  selected_genres :
     # 'Pop','Rock','Hip-Hop','Classical'
     max_tracks_each_genre = 1000 # choose how many songs you will 
     classical = dataset[dataset['genre_top'] == 'Classical'].head(max_tracks_each_genre)
     rock = dataset[dataset['genre_top'] == 'Rock'].head(max_tracks_each_genre)
     hip_hop = dataset[dataset['genre_top'] == 'Hip-Hop'].head(max_tracks_each_genre)
     pop = dataset[dataset['genre_top'] == 'Pop'].head(max_tracks_each_genre)
     # append each to our final dataset 
     final_dataset = pd.DataFrame()
     final_dataset = final_dataset.append([classical,hip_hop,rock,pop])

     return final_dataset

def extract_to_csv(df):

      df.to_csv(path_or_buf='./data/final.csv')

      return 

def merge_echonest(echo,tracks,features):
	echo_test=echo.loc[:,:'valence']
	echo_test=echo_test.merge(tracks[['track_id','genre_top']],on='track_id',how='inner')
	echo_test=echo_test.merge(features,on='track_id',how='inner')
	echo_test['genre_top'].value_counts()
	
	echo_test= echo_test.sample(frac=1).reset_index(drop=True)
     
	#  selected_genres :
	# 'Pop','Rock','Hip-Hop','Classical'
	max_tracks_each_genre = 450 # choose how many songs you will 
	genre1 = echo_test[echo_test['genre_top'] == 'Electronic'].head(max_tracks_each_genre)
	genre2 = echo_test[echo_test['genre_top'] == 'Rock'].head(max_tracks_each_genre)
	genre3 = echo_test[echo_test['genre_top'] == 'Hip-Hop'].head(max_tracks_each_genre)
	genre4 = echo_test[echo_test['genre_top'] == 'Folk']
	# append each to our final dataset 
     
	final = pd.DataFrame()
	final = final.append([genre1,genre2,genre3,genre4])
	
	return final
	


if __name__=='__main__':
     #set directory paths
     TRACK_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/tracks.csv'
     GENRE_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/genres.csv'
     FEATURE_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/features.csv'
     ECH0NEST_PATH = '/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata//echonest.csv'


     #csv to dataframe
     tracks = pd.read_csv(TRACK_PATH,header = 1 )
     genres = pd.read_csv(GENRE_PATH)
     features = pd.read_csv(FEATURE_PATH)
     echonest = pd.read_csv(ECH0NEST_PATH,header = 2)

     (tracks,echonest,features) = preprocess_csv_files(tracks,echonest,features)
     """make a dataset of 400 * 4 categories(POP,ROCK,CLASSICAL,HIP-HOP)"""
     # !!!!!!!!
     # just to find where are the mfcc mean in features df 
     p = pd.DataFrame()
     for column in features.columns:
         temp = features[column].iloc[0]
         #print(temp)
         if (temp == "mean"):
             p = p.join(features[column])
  
     c = pd.array(p.columns)
     #
     final = make_tracks_dataset(tracks)

     final.rename(columns={"zcr.2": "zcr"}, inplace=True)
     final.rename(columns={"spectral_rolloff.2": "spectral_rollof"}, inplace=True)
     final.rename(columns={"spectral_centroid.2": "spectral_centroid"}, inplace=True)
     print(final.head())
     #extract_to_csv(final)


