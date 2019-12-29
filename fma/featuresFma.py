import pandas as pd
features = pd.read_csv('features.csv')

features.rename(columns={ features.columns[0]: "track_id" }, inplace = True)

mfcc = features.loc[:,"mfcc":"mfcc.139"]
spectral_bandwith = features.loc[:,"spectral_centroid":"spectral_centroid.6"]
spectral_rollof   = features.loc[:,"spectral_rolloff":"spectral_rolloff.6"]
zcr   = features.loc[:,"zcr":"zcr.6"]
