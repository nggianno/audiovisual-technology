import pandas as pd
#features = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/features.csv')
echonest = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/echonest.csv',header = 1)

print(echonest.head(10))
#features.rename(columns={features.columns[0]: "track_id"},inplace=True)
#print(features.head(10))
print(echonest.loc[0])

#mfcc = features.loc[:,"mfcc":"mfcc.139"]
# spectral_centroid = pd.DataFrame(features.loc[:,"spectral_centroid":"spectral_centroid.6"])
# spectral_bandwidth = pd.DataFrame(features.loc[:,"spectral_bandwidth":"spectral_bandwidth.6"])
# spectral_rollof = pd.DataFrame(features.loc[:,"spectral_rolloff":"spectral_rolloff.6"])
# zcr = pd.DataFrame(features.loc[:,"zcr":"zcr.6"])

rmse = pd.DataFrame(features.loc[:,"rmse":"rmse.6"])
print(rmse.loc[0])
#print(mfcc)
#print(spectral_bandwidth.loc[0])
#print(spectral_rollof)
# avg_zcr = zcr['zcr.2']
# print(spectral_rollof.loc[0])
# print(spectral_bandwidth.loc[0])
# print(spectral_bandwidth.loc[0])
# print(avg_zcr)
