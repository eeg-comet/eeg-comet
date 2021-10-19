#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:31:52 2021

@author: amin
"""

import sys
import os
from fnmatch import fnmatch
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time


create_dataset = False

start = time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 100


classes = ['A', 'B', 'C', 'D', 'E', 'F']
raw_dataset_path = '/media/amin/Seagate Expansion Drive/AMIN/RS_EEG/RSEEG/img_clustering/classify_new/'
output_dataset_path = '/home/amin/Encfs/TMSEEG_DATA/EEG-Microstate-Feature-Extraction/microlabels_dataset/'
train_data_path = os.path.join(output_dataset_path, 'train')
validation_data_path = os.path.join(output_dataset_path, 'validation')
test_data_path = os.path.join(output_dataset_path, 'test')

# Generating the Dataset
if create_dataset:
    print('Creating the dataset ...')
    
    if not os.path.exists(output_dataset_path):
        os.rmdir(output_dataset_path)
        os.mkdir(output_dataset_path)
    
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)
    
    if not os.path.exists(validation_data_path):
        os.mkdir(validation_data_path)
    
    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)
    
    for C in classes:
        if not os.path.exists(os.path.join(train_data_path, C)):
            os.mkdir(os.path.join(train_data_path, C))
        if not os.path.exists(os.path.join(validation_data_path, C)):
            os.mkdir(os.path.join(validation_data_path, C))
        if not os.path.exists(os.path.join(test_data_path, C)):
            os.mkdir(os.path.join(test_data_path, C))
    
    extension = '*.png'
    list_images_A = []
    list_images_B = []
    list_images_C = []
    list_images_D = []
    list_images_E = []
    list_images_F = []
    
    for path, subdirs, files in os.walk(raw_dataset_path):
        for name in files:
            if fnmatch(name, extension):
                if path.endswith('A'):
                    list_images_A.append(os.path.join(path, name))
                if path.endswith('B'):
                    list_images_B.append(os.path.join(path, name))
                if path.endswith('C'):
                    list_images_C.append(os.path.join(path, name))
                if path.endswith('D'):
                    list_images_D.append(os.path.join(path, name))
                if path.endswith('E'):
                    list_images_E.append(os.path.join(path, name))
                if path.endswith('F'):
                    list_images_F.append(os.path.join(path, name))
    
    
    train_images_A = random.sample(list_images_A, int(len(list_images_A)*.7))
    other_images_A = list(set(list_images_A)^set(train_images_A))
    validation_images_A = random.sample(other_images_A, int(len(other_images_A)*.5))
    test_images_A = list(set(other_images_A)^set(validation_images_A))
    
    train_images_B = random.sample(list_images_B, int(len(list_images_B)*.7))
    other_images_B = list(set(list_images_B)^set(train_images_B))
    validation_images_B = random.sample(other_images_B, int(len(other_images_B)*.5))
    test_images_B = list(set(other_images_B)^set(validation_images_B))
    
    train_images_C = random.sample(list_images_C, int(len(list_images_C)*.7))
    other_images_C = list(set(list_images_C)^set(train_images_C))
    validation_images_C = random.sample(other_images_C, int(len(other_images_C)*.5))
    test_images_C = list(set(other_images_C)^set(validation_images_C))
    
    train_images_D = random.sample(list_images_D, int(len(list_images_D)*.7))
    other_images_D = list(set(list_images_D)^set(train_images_D))
    validation_images_D = random.sample(other_images_D, int(len(other_images_D)*.5))
    test_images_D = list(set(other_images_D)^set(validation_images_D))
    
    train_images_E = random.sample(list_images_E, int(len(list_images_E)*.7))
    other_images_E = list(set(list_images_E)^set(train_images_E))
    validation_images_E = random.sample(other_images_E, int(len(other_images_E)*.5))
    test_images_E = list(set(other_images_E)^set(validation_images_E))
    
    train_images_F = random.sample(list_images_F, int(len(list_images_F)*.7))
    other_images_F = list(set(list_images_F)^set(train_images_F))
    validation_images_F = random.sample(other_images_F, int(len(other_images_F)*.5))
    test_images_F = list(set(other_images_F)^set(validation_images_F))
    
    print('A')
    for file in train_images_A:
        shutil.copy(os.path.join(file), os.path.join(train_data_path, 'A'))
    for file in validation_images_A:
        shutil.copy(os.path.join(file), os.path.join(validation_data_path, 'A'))
    for file in test_images_A:
        shutil.copy(os.path.join(file), os.path.join(test_data_path, 'A'))
    
    print('B')
    for file in train_images_B:
        shutil.copy(os.path.join(file), os.path.join(train_data_path, 'B'))
    for file in validation_images_B:
        shutil.copy(os.path.join(file), os.path.join(validation_data_path, 'B'))
    for file in test_images_B:
        shutil.copy(os.path.join(file), os.path.join(test_data_path, 'B'))
        
    print('C')
    for file in train_images_C:
        shutil.copy(os.path.join(file), os.path.join(train_data_path, 'C'))
    for file in validation_images_C:
        shutil.copy(os.path.join(file), os.path.join(validation_data_path, 'C'))
    for file in test_images_C:
        shutil.copy(os.path.join(file), os.path.join(test_data_path, 'C'))
    
    print('D')
    for file in train_images_D:
        shutil.copy(os.path.join(file), os.path.join(train_data_path, 'D'))
    for file in validation_images_D:
        shutil.copy(os.path.join(file), os.path.join(validation_data_path, 'D'))
    for file in test_images_D:
        shutil.copy(os.path.join(file), os.path.join(test_data_path, 'D'))
    
    print('E')
    for file in train_images_E:
        shutil.copy(os.path.join(file), os.path.join(train_data_path, 'E'))
    for file in validation_images_E:
        shutil.copy(os.path.join(file), os.path.join(validation_data_path, 'E'))
    for file in test_images_E:
        shutil.copy(os.path.join(file), os.path.join(test_data_path, 'E'))
    
    print('F')
    for file in train_images_F:
        shutil.copy(os.path.join(file), os.path.join(train_data_path, 'F'))
    for file in validation_images_F:
        shutil.copy(os.path.join(file), os.path.join(validation_data_path, 'F'))
    for file in test_images_F:
        shutil.copy(os.path.join(file), os.path.join(test_data_path, 'F'))


"""
Parameters
"""
img_width, img_height = 128, 128
batch_size = 16
samples_per_epoch = 16
validation_steps = 8
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 16
nb_filters4 = 16
conv1_size = 7
conv2_size = 5
conv3_size = 3
conv4_size = 3
pool_size = 2
classes_num = 6
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