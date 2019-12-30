import pandas as pd
features = pd.read_csv('/home/nick/Desktop/yliko_sxolhs/AudioVisual Technology/fma_metadata/features.csv')

features.rename(columns={features.columns[0]: "track_id"},inplace=True)

mfcc = features.loc[:,"mfcc":"mfcc.139"]
spectral_bandwidth = features.loc[:,"spectral_centroid":"spectral_centroid.6"]
spectral_rollof = features.loc[:,"spectral_rolloff":"spectral_rolloff.6"]
zcr = features.loc[:,"zcr":"zcr.6"]

print(mfcc)
print(spectral_bandwidth)
print(spectral_rollof)
print(zcr)
