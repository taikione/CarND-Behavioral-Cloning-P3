import os
import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

samples = []
with open('mydata/driving_log.csv') as csvfile, open('mydata2/driving_log.csv') as csvfile2:
    for f in [csvfile, csvfile2]:
        reader = csv.reader(f)
        # fieldnames = next(reader)
        for line in reader:
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
BATCH_SIZE = 32
EPOCHS=2

def get_callbacks():
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    checkpointer = ModelCheckpoint(filepath="models/model_{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
    return [earlystop, reduce_lr, checkpointer]

def generator(samples, batch_size=BATCH_SIZE, augmentation=0):
    num_samples = len(samples)
    if augmentation == 1:
        batch_size = int(batch_size/2)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = 'data/'+batch_sample[0]
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # data augmentation added flipped images
                if augmentation == 1:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, augmentation=0)
validation_generator = generator(validation_samples, augmentation=0)

ch, row, col = 3, 90, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
# Resized 25x80
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

model.add(Conv2D(24, (5, 5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(36, (5, 5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))

adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
model.compile(loss='mse', optimizer=adam)
checkpointer = get_callbacks()
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, epochs=EPOCHS, callbacks=checkpointer, validation_data=validation_generator, validation_steps=len(validation_samples))
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

