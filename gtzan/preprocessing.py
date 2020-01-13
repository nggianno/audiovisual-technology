import os
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


DIR = '/home/user/Downloads/genres'


def load_data():
    label = []
    zero_crossings = []
    spectral_centroids = []
    mfccs = []
    mels = []
    list = ['classical','hiphop','pop','rock']
    for file in list:
        file_path = os.path.join(DIR, file)
        i=0
        for wav in os.listdir(file_path):
            wav_path = os.path.join(file_path,wav)
            #add the songs and the labels to lists
            data, sampling_rate = librosa.load(wav_path)
            label.append(file)

            #calculate zero_crossings
            zeros = librosa.zero_crossings(data, pad=False)
            zero_crossings.append(sum(zeros))

            #calculate spectral_centroid -- TAKES MUCH TIME
            sc = librosa.feature.spectral_centroid(data, sr=sampling_rate)[0]
            spectral_centroids.append(sc)

            #calculate the Mel frequency cepstral coefficients
            mf = librosa.feature.mfcc(data, sr=sampling_rate)
            mfccs.append(mf)

            #calcaulate the spectrogram for every track
            ps = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=128)
            # save the spectrograms as png files
            ps = librosa.power_to_db(ps, ref=np.max)
            librosa.display.specshow(ps, x_axis='time', y_axis='mel', sr=sampling_rate, fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.savefig('/home/user/genres/spectogram' +str(i) + '.png')
            i+=1
            # save the spectrograms as csv files
            #mels.append(np.array(ps))
            #mels = pd.DataFrame(ps)
            #mels.to_csv(path_or_buf='/home/rigas/Downloads/genres/hiphop_spectogram/spectrogram' +'hiphop' +str(i) + '.csv')
            
    return label, zero_crossings,spectral_centroids,mfccs,mels

(label, zero_crossings,spectral_centroids,mfccs,mels) = load_data()

#save the labels to a txt file
with open("labels.txt", "w") as f:
    for item in label:
        f.write("%s\n" % item)
with open("zero_crossings.txt", "w") as f:
    for item in zero_crossings:
        f.write("%s\n" % item)
