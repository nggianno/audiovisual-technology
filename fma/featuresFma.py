import pandas as pd
features = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/features.csv')

features.rename(columns={features.columns[0]: "track_id"},inplace=True)
print(features.head(10))

#mfcc = features.loc[:,"mfcc":"mfcc.139"]
spectral_bandwidth = pd.DataFrame(features.loc[:,"spectral_centroid":"spectral_centroid.6"])
spectral_rollof = pd.DataFrame(features.loc[:,"spectral_rolloff":"spectral_rolloff.6"])
zcr = pd.DataFrame(features.loc[:,"zcr":"zcr.6"])


#print(mfcc)
#print(spectral_bandwidth.loc[0])
#print(spectral_rollof)
avg_zcr = zcr['zcr.2']
print(spectral_rollof.loc[0])
print(spectral_bandwidth.loc[0])
print(zcr.loc[0])
print(avg_zcr)
