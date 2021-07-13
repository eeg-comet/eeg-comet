#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:31:52 2021

@author: amin
"""

import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time



start = time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 20

'''
# Unzipping Dataset
if not os.path.exists('dataset'):
    fh = open('dataset.zip', 'rb')
    zf = zipfile.ZipFile(fh)
    uncompress_size = sum((file.file_size for file in zf.infolist()))
    extracted_size = 0
    print('Unzipping ...')
    for file in zf.infolist():
        extracted_size += file.file_size
        #zf.extract(member=file, path='all_ecg_data')
        zf.extract(member=file)
        progress = extracted_size * 100/uncompress_size
        if round(progress)%5==0:
            print("%s %%\r %" , round(progress), end="\r")
'''

train_data_path = 'img_dataset/train'
validation_data_path = 'img_dataset/validation'

"""
Parameters
"""
img_width, img_height = 128, 128
batch_size = 8
samples_per_epoch = 16
validation_steps = 16
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 16
nb_filters4 = 16
conv1_size = 7
conv2_size = 5
conv3_size = 3
conv4_size = 3
pool_size = 2
classes_num = 5
lr = 0.001

model = Sequential()

model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding ="SAME", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding ="SAME"))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters3, conv3_size, conv3_size, padding ="SAME"))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters4, conv4_size, conv4_size, padding ="SAME"))
model.add(Activation("relu"))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(classes_num, activation='softmax'))

model.summary()

print("\n*** Weights Shape ***	\n")
for i, layer in enumerate(model.layers):
    if len(layer.get_weights()) > 0:
       W, b = layer.get_weights()
       print("Layer", i, "\t", layer.name, "\t\t", W.shape, "\t", b.shape)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')

#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")