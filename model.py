import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    fieldnames = next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/'+batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

ch, row, col = 3, 90, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
# Resized 25x80
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(MaxPooling2D(pool_size=(4,4), padding='valid'))

model.add(Conv2D(24, (5, 5), activation='elu', padding='valid'))
model.add(Conv2D(36, (5, 5), activation='elu', padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5), activation='elu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='elu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='elu', padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), epochs=10, validation_data=validation_generator, validation_steps=len(validation_samples))
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('imgs/loss.png')
