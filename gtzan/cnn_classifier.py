from keras.models import Sequential, K
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.layers. normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import array, genfromtxt
import pickle
import os
import pandas as pd
K.clear_session()

DIR = '/home/rigas/Downloads/genres/spectograms'

X_train = []
y_train = []
data = []
for file in os.listdir(DIR):
        file_path = os.path.join(DIR, file)
        for csv in os.listdir(file_path):
                csv_path = os.path.join(file_path, csv)
                #data = pd.read_csv(csv_path, header=None)
                data = genfromtxt(csv_path, delimiter=',')
                X_train.append(data)
                y_train.append(file)
#X_train = np.array(X_train)
#X_train = X_train.astype('float32')
print(np.shape(X_train))
print(np.shape(data))


# Building the model
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', padding="same",input_shape=(IMG_HEIGHT,IMG_WIDTH,  1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), padding="same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), padding="same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Dense(4, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train,validation_data=(x_test,y_test), batch_size=256, epochs=60)

model.save("model.h5")
with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
print("Saved model to disk")

