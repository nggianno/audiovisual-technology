import IPython.display as ipd
import os
import pandas as pd
import librosa.display
import glob
import matplotlib.pyplot as plt

DIR = '/home/user/genres'

#play the wav file
#ipd.Audio('/home/user/genres/blues/blues.00000.wav')


#plot the wav file
#x, sr = librosa.load('/home/user/blues/blues.00000.wav')
#plt.figure(figsize=(12, 4))
#librosa.display.waveplot(data, sr=sampling_rate)
#plt.show()

def load_data():
    label = []
    zero_crossings = []
    #spectral_centroids = []
    mfccs = []

    for file in os.listdir(DIR):
        file_path = os.path.join(DIR, file)
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

    return label, zero_crossings, sc, mfccs

(label, zero_crossings, sc, mfccs) = load_data()
